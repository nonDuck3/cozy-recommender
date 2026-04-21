from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.client import User
from diagrams.programming.language import Python
from diagrams.aws.ml import Sagemaker 
from diagrams.onprem.network import Internet
from diagrams.generic.storage import Storage
from diagrams.aws.compute import Lambda 
from diagrams.onprem.compute import Server

GRAPH_ATTR = {
    "fontsize": "22",
    "fontname": "Sans-Serif bold",
    "bgcolor": "transparent",
    "pad": "0.3",
    "splines": "spline",
    "ranksep": "1.2", 
    "nodesep": "0.8",
}

NODE_ATTR = {
    "fontsize": "14",
    "fontname": "Sans-Serif bold",
}

CLUSTER_ATTR = {
    "fontsize": "20",
    "fontname": "Sans-Serif bold",
    "margin": "40.0", 
}

def main():
    with Diagram(
        "🧥❄️ Winter Apparel Recommender System",
        show=False,
        filename="docs/workflow",
        graph_attr=GRAPH_ATTR,
        node_attr=NODE_ATTR,
    ):
        with Cluster("🖥️ Local / Offline Pipeline", graph_attr=CLUSTER_ATTR):
            site_1 = Internet("🌐 Store 1 site")
            scrape_1 = Python("🕷️ Web Scraper")
            csv_1 = Storage("📄 store_1_items.csv")
            siglip_1 = Sagemaker("🧠 SigLIP (local)\nEncode + Embed Data")
            sanity_1 = Lambda("Σ Cosine similarity")

            site_2 = Internet("🌐 Store 2 site")
            scrape_2 = Python("🕷️ Web Scraper")
            csv_2 = Storage("📄 store_2_items.csv")
            siglip_2 = Sagemaker("🧠 SigLIP (local)\nEncode + Embed Data")
            sanity_2 = Lambda("Σ Cosine similarity")

            site_n = Internet("🌐 Store N site")
            scrape_n = Python("🕷️ Web Scraper")
            csv_n = Storage("📄 store_n_items.csv")
            siglip_n = Sagemaker("🧠 SigLIP (local)\nEncode + Embed Data")
            sanity_n = Lambda("Σ Cosine similarity")

            master = Storage("💾 Master embeddings\n(.pkl)")

            site_n >> scrape_n >> csv_n >> siglip_n >> sanity_n >> master
            site_2 >> scrape_2 >> csv_2 >> siglip_2 >> sanity_2 >> master
            site_1 >> scrape_1 >> csv_1 >> siglip_1 >> sanity_1 >> master

        with Cluster("☁️ Streamlit Cloud — Real-Time App", graph_attr=CLUSTER_ATTR):
            user = User("👤 User")
            ui = Internet("🧾 Streamlit UI\n(text + budget)")
            encode_query = Sagemaker("🧠 SigLIP\nEncode user text")
            similarity = Lambda("Σ Cosine Similarity") 
            filter_price = Server("🔍 Filter Logic")
            output = Storage("📊 Display top 10 Results") 
            user >> ui >> encode_query >> similarity >> filter_price >> output

        # Bridge
        master >> Edge(label="LOAD EMBEDDINGS", fontname="Sans-Serif bold", fontsize="20", color="#22314a") >> similarity

if __name__ == "__main__":
    main()
    