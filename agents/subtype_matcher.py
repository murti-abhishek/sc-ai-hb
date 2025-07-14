# agents/subtype_matcher.py

from dotenv import load_dotenv
import os
import json
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=openai_api_key)

# Define prompt template
prompt = PromptTemplate(
    input_variables=["cluster_id", "gene_list"],
    template="""
You are a liver cancer expert analyzing single-cell transcriptomics data from a hepatoblastoma tumor.

The following genes are the most upregulated in cluster {cluster_id}:
{gene_list}

Based on known hepatoblastoma subtypes (fetal, embryonal, macrotrabecular, mixed), which subtype is this cluster most likely to represent? Justify your reasoning based on gene functions and tumor biology literature. Provide a clear subtype prediction and explanation.
"""
)

# Use new-style chain with RunnableSequence
chain = prompt | llm

# Input/output paths
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

            hypothesis = {
                "cluster": cluster_id,
                "top_genes": genes,
                "llm_response": response.content.strip()
            }

            output_file = OUTPUT_DIR / f"cluster_{cluster_id}_hypothesis.json"
            with open(output_file, "w") as out:
                json.dump(hypothesis, out, indent=2)

            print(f"Saved hypothesis for cluster {cluster_id}.")

        except Exception as e:
            print(f"Error processing cluster {cluster_id}: {e}")

if __name__ == "__main__":
    main()
