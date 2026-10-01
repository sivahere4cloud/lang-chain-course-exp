from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = "medium-analyzer"
DIMENSIONS = 1536
metric = "cosine"


CLOUD = "aws"
REGION = "us-east-1"

def main() -> None:
    pc = Pinecone()

    index_exists = pc.has_index(INDEX_NAME)
    print("Index already exists:",index_exists)

    if not index_exists:
        print("Creating index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSIONS,
            metric=metric,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION)
        )
    print("Index created successfully")

    description = pc.describe_index(INDEX_NAME)
    print("Index description:",description)
    print("Name:",description.name)
    print("Metric:",description.metric)
    print("Ready:", description.status.ready)
 


if __name__ == "__main__":
    main()