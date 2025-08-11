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
llm = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=openai_api_key)

# Prompt template with risk scores explanation and request for interpretation
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

The following normalized risk scores (scaled 0 to 1) have been computed for this cluster:
- Proliferation score (0 = low proliferation, 1 = high proliferation): {proliferation_score}
- Stemness score (0 = low stemness, 1 = high stemness): {stemness_score}
- Immune infiltration score (0 = low immune infiltration, 1 = high infiltration): {immune_score}
- Prognostic signature score (0 = low risk, 1 = high risk): {prognostic_score}

Based on these gene markers and risk scores, and known hepatoblastoma subtypes (fetal, embryonal, macrotrabecular, mixed), please return a JSON object with the following structure. In "SupportingEvidence", please interpret these scores, indicating if they are high, low, or intermediate, and what that implies for the subtype assignment and tumor biology:

{{
  "Cluster": "<cluster ID>",
  "CandidateSubtype": "<one of fetal, embryonal, macrotrabecular, mixed>",
  "TopGenes": [<list of top marker genes>],
  "ProliferationScore": "{proliferation_score}",
  "StemnessScore": "{stemness_score}",
  "ImmuneScore": "{immune_score}",
  "PrognosticScore": "{prognostic_score}",
  "SupportingEvidence": [<short bullet points explaining reasoning including score interpretation>],
  "SuggestedExperiments": [<short list of follow-up biological experiments>]
}}

Be concise but accurate. Use known literature and tumor biology concepts. Return **only the JSON object**.
"""
)

# Use the new LangChain Runnable interface
chain: RunnableSequence = prompt | llm

# Input and output locations
INPUT_FILE = "outputs/top_genes_by_cluster.json"
OUTPUT_DIR = Path("outputs/hypotheses/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load scaled risk scores CSV
RISK_SCORES_FILE = "outputs/cluster_risk_scores_scaled.csv"
risk_scores_df = pd.read_csv(RISK_SCORES_FILE, index_col=0)

# Convert scaled risk scores to dictionaries for lookup
proliferation_scores = risk_scores_df['proliferation_score_scaled'].to_dict()
stemness_scores = risk_scores_df['stemness_score_scaled'].to_dict()
immune_scores = risk_scores_df['immune_score_scaled'].to_dict()
prognostic_scores = risk_scores_df['prognostic_score_scaled'].to_dict()

def main():
    with open(INPUT_FILE, "r") as f:
        cluster_genes = json.load(f)

    for cluster_id, genes in cluster_genes.items():
        print(f"Processing cluster {cluster_id}...")

        try:
            input_vars = {
                "cluster_id": cluster_id,
                "gene_list": ", ".join(genes),
                "proliferation_score": f"{proliferation_scores.get(cluster_id, 0):.3f}",
                "stemness_score": f"{stemness_scores.get(cluster_id, 0):.3f}",
                "immune_score": f"{immune_scores.get(cluster_id, 0):.3f}",
                "prognostic_score": f"{prognostic_scores.get(cluster_id, 0):.3f}",
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
