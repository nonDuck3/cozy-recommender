import argparse
from web_scraper import WebScraper
from vector_encoder import encode_images_with_text, combine_all_embeddings, compute_cosine_similarity

def main():
    """
    Orchestrates the web scraping process based on command-line arguments.

    This function initializes the argument parser to capture user inputs (such as 
    page URLs, store names, and webdriver). It then instantiates 
    the web scraper class, executes the scraping logic to collect image and 
    description data, and handles the final data merging and export.

    The workflow includes:
        1. Parsing CLI arguments via argparse.
        2. Initializing the scraper with the provided JSON configuration.
        3. Iteratively scraping pages to build image and description DataFrames.
        4. Merging datasets, cleaning duplicates, and resetting IDs.
        5. Exporting the final cleaned dataset to a CSV file.
    """
    parser = argparse.ArgumentParser(description="Web scraping tool for extracting items, prices and images of clothes from Australian stores.")

    parser.add_argument('page_url', help="URL of clothing store to scrape")
    parser.add_argument('store', help="Name of store")
    parser.add_argument('driver', help="File path of webdriver")
    parser.add_argument('siglip_path', help="Folder path that stores siglip model")
    parser.add_argument('user_style', help="Query string containing user preference")
    parser.add_argument('--df_col', help="Column name used for merging final dataframe", default="Id")
    parser.add_argument('--max_price', help="Max price of item user will pay", default=100000)

    args = parser.parse_args()

    folder_name = "./data/embeddings/*.pkl"
    file = "./data/embeddings/all_embeddings.pkl"

    scraper = WebScraper(page_url=f"{args.page_url}", store_name=f"{args.store}", driver=f"{args.driver}")
    image_df, description_df = scraper.scrape_infinite_scroll_page()
    image_df = scraper.get_image_df(image_df=image_df)
    description_df = scraper.get_description_df(description_df=description_df)
    scraped_data_df = scraper.merge_export_to_csv(image_df=image_df, description_df=description_df, merge_col=f"{args.df_col}")

    encode_images_with_text(local_path=f"{args.siglip_path}", scraped_df=scraped_data_df)
    combine_all_embeddings(folder=folder_name, file_name=file)
    compute_cosine_similarity(file_path=f"{args.siglip_path}", user_pref=f"{args.user_style}", file=file, max_price=args.max_price)

if __name__ == "__main__":
    main()
