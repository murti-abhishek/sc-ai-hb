from dotenv import load_dotenv
import os
import json
import ast
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

# Load environment variables from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChatOpenAI model via LangChain
llm = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=openai_api_key)

# Prompt template with structured output request
prompt = PromptTemplate(
    input_variables=["cluster_id", "gene_list"],
    template="""
You are a liver cancer expert analyzing single-cell transcriptomics data from a hepatoblastoma tumor.

The following genes are the most upregulated in cluster {cluster_id}:
{gene_list}

Based on known hepatoblastoma subtypes (fetal, embryonal, macrotrabecular, mixed), please return a JSON object with the following structure:

{{
  "Cluster": "<cluster ID>",
  "CandidateSubtype": "<one of fetal, embryonal, macrotrabecular, mixed>",
  "TopGenes": [<list of top marker genes>],
  "SupportingEvidence": [<short bullet points explaining reasoning>],
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

def main():
    with open(INPUT_FILE, "r") as f:
        cluster_genes = json.load(f)

    for cluster_id, genes in cluster_genes.items():
        print(f"Processing cluster {cluster_id}...")

        try:
            input_vars = {
                "cluster_id": cluster_id,
                "gene_list": ", ".join(genes)
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

            # Save the structured hypothesis
            output_file = OUTPUT_DIR / f"cluster_{cluster_id}_hypothesis.json"
            with open(output_file, "w") as out:
                json.dump(hypothesis_data, out, indent=2)

            print(f"✅ Saved hypothesis for cluster {cluster_id}.")

        except Exception as e:
            print(f"❌ Error processing cluster {cluster_id}: {e}")

if __name__ == "__main__":
    main()
