from dotenv import load_dotenv
from tavily import Client, TavilyClient

load_dotenv()

client = TavilyClient()
response = client.search(query="Current weather in London") 


print("Type:", type(response))
print("Keys:", response.keys())
print("Number of results:", len(response["results"]))

first_result = response["results"][0]
print("Keys of one result:", first_result.keys())

for result in response["results"]:
    print("-----")
    print("Title:", result["title"])
    print("URL:", result["url"])