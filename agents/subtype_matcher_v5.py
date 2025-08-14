from dotenv import load_dotenv
import os
import json
import ast
from pathlib import Path
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

# Load environment variables from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChatOpenAI model via LangChain
llm = ChatOpenAI(model="gpt-5", openai_api_key=openai_api_key)

# Prompt template: scores have interpretation next to numeric values
prompt = PromptTemplate(
    input_variables=[
        "cluster_id",
        "gene_list",
        "proliferation_score",
        "stemness_score",
        "immune_score",
        "prognostic_score",
    ],
    template="""
You are a liver cancer expert analyzing single-cell transcriptomics data from a hepatoblastoma tumor.

The following genes are the most upregulated in cluster {cluster_id}:
{gene_list}

The following normalized risk scores (scaled 0 to 1) have been computed for this cluster, with interpretations in parentheses:
- Proliferation score: {proliferation_score}
- Stemness score: {stemness_score}
- Immune infiltration score: {immune_score}
- Prognostic signature score: {prognostic_score}

Based on these gene markers and known hepatoblastoma subtypes (fetal, embryonal, macrotrabecular, mixed), please return a JSON object with the following structure:

{{
  "Cluster": "<cluster ID>",
  "CandidateSubtype": "<one of fetal, embryonal, macrotrabecular, mixed>",
  "TopGenes": [<list of top marker genes>],
  "ProliferationScore": "{proliferation_score}",
  "StemnessScore": "{stemness_score}",
  "ImmuneScore": "{immune_score}",
  "PrognosticScore": "{prognostic_score}",
  "SupportingEvidence": [<short bullet points explaining reasoning based on gene markers and tumor biology>],
  "SuggestedExperiments": [<short list of follow-up biological experiments>]
}}

Do NOT interpret the scores in SupportingEvidence — just include them directly in the respective score fields above. Be concise but accurate, and return **only the JSON object**.
"""
)

# Use the new LangChain Runnable interface
chain: RunnableSequence = prompt | llm

# Input and output locations
INPUT_FILE = "outputs/top_genes_by_leiden_cluster.json"
OUTPUT_DIR = Path("outputs/hypotheses/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load scaled risk scores CSV
RISK_SCORES_FILE = "outputs/leiden_cluster_risk_scores_scaled.csv"
risk_scores_df = pd.read_csv(RISK_SCORES_FILE)

# Ensure cluster IDs are the index and strings to match JSON
if 'cluster' in risk_scores_df.columns:
    risk_scores_df.set_index('cluster', inplace=True)
risk_scores_df.index = risk_scores_df.index.astype(str)

# Helper to label score
def interpret_score(value):
    if value >= 0.67:
        label = "high"
    elif value >= 0.34:
        label = "intermediate"
    else:
        label = "low"
    return f"{value:.3f} ({label})"

# Convert scaled risk scores to dictionaries with interpretation
proliferation_scores = {k: interpret_score(v) for k, v in risk_scores_df['proliferation_score_scaled'].to_dict().items()}
stemness_scores = {k: interpret_score(v) for k, v in risk_scores_df['stemness_score_scaled'].to_dict().items()}
immune_scores = {k: interpret_score(v) for k, v in risk_scores_df['immune_score_scaled'].to_dict().items()}
prognostic_scores = {k: interpret_score(v) for k, v in risk_scores_df['prognostic_score_scaled'].to_dict().items()}

def main():
    with open(INPUT_FILE, "r") as f:
        cluster_genes = json.load(f)

    for cluster_id, genes in cluster_genes.items():
        print(f"Processing cluster {cluster_id}...")

        try:
            # Debug print to check if scores exist
            print(
                f"Cluster {cluster_id} scores -> "
                f"Proliferation: {proliferation_scores.get(cluster_id)}, "
                f"Stemness: {stemness_scores.get(cluster_id)}, "
                f"Immune: {immune_scores.get(cluster_id)}, "
                f"Prognostic: {prognostic_scores.get(cluster_id)}"
            )

            input_vars = {
                "cluster_id": cluster_id,
                "gene_list": ", ".join(genes),
                "proliferation_score": proliferation_scores.get(cluster_id, "0.000 (low)"),
                "stemness_score": stemness_scores.get(cluster_id, "0.000 (low)"),
                "immune_score": immune_scores.get(cluster_id, "0.000 (low)"),
                "prognostic_score": prognostic_scores.get(cluster_id, "0.000 (low)"),
            }

            response = chain.invoke(input_vars)
            raw_output = response.content.strip()

            try:
                hypothesis_data = ast.literal_eval(raw_output)
            except Exception:
                print(f"⚠️ Warning: Could not parse JSON for cluster {cluster_id}. Saving raw response.")
                hypothesis_data = {
                    "Cluster": cluster_id,
                    "TopGenes": genes,
                    "RawResponse": raw_output
                }

            output_file = OUTPUT_DIR / f"cluster_{cluster_id}_hypothesis.json"
            with open(output_file, "w") as out:
                json.dump(hypothesis_data, out, indent=2)

            print(f"✅ Saved hypothesis for cluster {cluster_id}.")

        except Exception as e:
            print(f"❌ Error processing cluster {cluster_id}: {e}")

if __name__ == "__main__":
    main()
