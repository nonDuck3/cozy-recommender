from selenium import webdriver
from selenium.webdriver.edge.service import Service
import time

from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.edge.options import Options
import json
import re

class WebScraper:
    """
    A web scraping utility for extracting and storing data from web pages.

    This class orchestrates a Selenium WebDriver to navigate to a specific URL,
    extracts and cleans the required data, and writes the contents as a Python dataframe to csv.
    """

    def __init__(self, page_url: str, store_name: str, driver: str):
        """
        Initializes the WebScraper with connection details and dependencies.

        Args:
            page_url (str): The full URL of the webpage to scrape.
            store_name (str): The clothig store name to scrape from.
            driver (selenium.webdriver): A configured Selenium WebDriver 
                (e.g., Edge, Chrome, Firefox).
        """
        self.page_url = page_url
        self.driver = driver
        self.store_name = store_name.lower()
        self.json_config = self.read_json_file()

    def read_json_file(self):
        """
        Reads and parses a JSON file from the local file system. 

        This method opens the specified file and attempts to decode the JSON 
        content into a Python dictionary or list.

        Returns:
            dict|list: The parsed JSON content, typically a dictionary or 
                a list of dictionaries.
        """
        with open(f"./configs/{self.store_name}.json") as f:
            config = json.load(f)
            return config

    def scrape_infinite_scroll_page(self):
        """
        Initializes the browser automation, HTML parser, and data storage structures. 
        
        This method performs the heavy lifting required before scraping begins:
        1. Launches the Selenium WebDriver (Edge).
        2. Instantiates the BeautifulSoup parser for DOM traversal.
        3. Initializes an empty pandas DataFrame with predefined columns to standardize data collection.

        Returns:
            image_df (pd.DataFrame): empty Python dataframe to store image urls.
            description_df (pd.DataFrame): empty Python dataframe to store clothing names and prices.
        """
        # run browser quicker in headless mode and ignore SSL warnings
        browser_options = Options()
        browser_options.add_argument("--headless")
        browser_options.add_argument('--ignore-certificate-errors')
        browser_options.add_argument('--ignore-ssl-errors')

        service = Service(self.driver)
        driver = webdriver.Edge(service=service, options=browser_options)
        driver.get(self.page_url)

        last_height = driver.execute_script('return document.body.scrollHeight')

        driver.set_window_size(1920, 10800) 

        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight + 100);")
            time.sleep(2)
            
            driver.execute_script("window.scrollBy(0, -300);")
            time.sleep(1)
            driver.execute_script("window.scrollBy(0, 300);")
            time.sleep(2) 

            new_height = driver.execute_script("return document.body.scrollHeight")
            
            if new_height == last_height:
                print("End of page has been reached.")
                break
                
            last_height = new_height

        self.soup = BeautifulSoup(driver.page_source, 'html.parser')

        image_df = pd.DataFrame(columns=['Id', 'Image Link'])
        description_df = pd.DataFrame(columns=['Id', 'Clothing', 'Price'])

        return image_df, description_df

    def extract_html_tags(self, element, steps):
        """
        Extracts a specific attribute value from a BeautifulSoup element. 

        This method acts as a dynamic extractor, taking a single HTML element (usually from a 'find_all' loop)
        and retrieving the value of a targeted attribute such as 'src', 'href', or 'style'.

        Args:
            element (bs4.element.Tag): The HTML tag object currently being 
                processed in the iteration.
            steps (str): The specific attribute key to retrieve from the tag 
                (e.g., "src" for images, "href" for links, or "style").

        Returns:
            current (str | None): The value of the requested attribute if it exists; 
                returns None or an empty string if the attribute is missing 
                to prevent the scraper from crashing.
        """
        current = element
        for step in steps:
            if not current:
                return None
            
            tag = step.get("tag")
            class_names = step.get("class")
            
            if tag:
                current = current.find(tag, class_=class_names)

            if not current:
                return None

            action = step.get("action")
            if action == "get":
                attr_name = step.get("attr")
                value = current.get(attr_name)
                if attr_name == "style" and value:
                    match = re.search(r'url\(["\']?(.*?)["\']?\)', value)
                    if match:
                        extracted_url = match.group(1).strip()
                        return extracted_url
                return value
                
            if action == "text":
                return current.get_text(strip=True)
            
        return current

    def get_description_df(self, description_df: pd.DataFrame):
        """
        Parses HTML content to extract clothing details and updates the description DataFrame. 
        
        This method iterates through HTML tags identified by the configuration settings. 
        For each element, it attempts to extract clothing names and prices, appending successful
        extractions to the provided DataFrame with a unique ID.

        Args:
            description_df (pd.DataFrame): The existing DataFrame to which the 
                extracted clothing data (ID, name, and price) will be appended.

        Returns:
            description_df (pd.DataFrame): The updated DataFrame containing the newly scraped 
                clothing descriptions and prices.
        """
        text_cards = self.soup.find_all(
            self.json_config["container"]["tag"], 
            class_=self.json_config["container"]["class"]
        )

        for text_id, card in enumerate(text_cards, start=1):
            try:
                clothing = self.extract_html_tags(card, self.json_config["fields"]["clothing"])
                price = self.extract_html_tags(card, self.json_config["fields"]["price"])
            except KeyError:
                print("Clothing/Price field missing in json config file!")

            if clothing and price:
                description_df = self.concat_description_df(description_df=description_df, id=text_id, clothing=clothing, price=price)

        return description_df

    def get_image_df(self, image_df: pd.DataFrame):
        """
        Extracts image metadata from HTML tags and populates a DataFrame.

        This method scans the current BeautifulSoup object for image containers 
        defined in the configuration. It extracts the image URL and alternative 
        text (alt text) for each card found, then appends this data to the 
        provided DataFrame.

        Args:
            image_df (pd.DataFrame): The DataFrame to which the extracted image 
                data (ID, image link, and alt text) will be appended.

        Returns:
            image_df (pd.DataFrame): The updated DataFrame containing the scraped image 
                information.
        """
        image_cards = self.soup.find_all(
            self.json_config["image_container"]["tag"], 
            class_=self.json_config["image_container"]["class"]
        )
    
        for image_id, card in enumerate(image_cards, start=1):
            image_url = ""
            image_alt = ""

            try:
                image_url = self.extract_html_tags(card, self.json_config["fields"]["image"])
            except KeyError:
                print("Image field missing in json config file!")

            try:
                image_alt = self.extract_html_tags(card, self.json_config["fields"]["image_alt"])
            except KeyError:
                pass

            if image_url:
                image_df = self.concat_image_df(
                    image_df=image_df, 
                    id=image_id, 
                    image_link=image_url, 
                    image_alt=image_alt
                )

        return image_df

    def concat_image_df(self, image_df: pd.DataFrame, id: int, image_link: str, image_alt: str =""):
        """
        Appends a new image record to the DataFrame and removes empty attribute columns. 
        
        This method creates a new record from the provided image metadata, 
        concatenates it with the existing DataFrame, and performs a cleanup step by dropping 
        any columns that consist entirely of empty strings ("").

        Args:
            image_df (pd.DataFrame): The current DataFrame containing image records.
            id (int): The unique identifier for the specific image or parent record.
            image_link (str): The direct URL or source path of the image.
            image_alt (str, optional): The alternative text description of the image. 
                Defaults to an empty string ("").

        Returns:
            image_df (pd.DataFrame): The updated DataFrame including the new record, 
                minus any columns that contained no data across all rows.
        """
        new_row = pd.DataFrame([{'Id': id, 'Clothing': image_alt, 'Image Link': image_link}])
        new_row = new_row.replace("", None).dropna(axis=1)
        image_df = pd.concat([image_df, new_row], ignore_index=True)
        return image_df

    def concat_description_df(self, description_df: pd.DataFrame, id: int, clothing: str, price: str):
        """
        Appends clothing and price data to the description tracking DataFrame. 
        
        This method takes a unique identifier and the associated price of a clothing item, 
        creates a new entry, and merges it with the existing description DataFrame using concatenation.

        Args:
            description_df (pd.DataFrame): The existing DataFrame containing 
                product descriptions.
            id (int): The unique identifier for the specific clothing item.
            clothing (str): The name of the piece of clothing.
            price (str): The price extracted from the web page (e.g. $29.99).

        Returns:
            description_df (pd.DataFrame): The updated DataFrame including the new records.
        """
        description_df = pd.concat([description_df, pd.DataFrame([{'Id': id,'Clothing': clothing, 'Price': price}])], ignore_index=True)
        return description_df

    def merge_drop_duplicates(self, image_df: pd.DataFrame, description_df: pd.DataFrame, merge_col: str):
        """
        Merges image and description DataFrames and removes duplicate image links. 
        
        This method performs an inner join between the two provided DataFrames. 
        After merging, it identifies records with duplicate values in the 'Image Link'
        column and keeps only the first occurrence.

        Args:
            image_df (pd.DataFrame): DataFrame containing image id and the 'Image Link' column.
            description_df (pd.DataFrame): DataFrame containing id, clothing, and price columns and a common key for merging.
            merge_col (str): The column of the DataFrame to merge on.
        Returns:
            combined_df (pd.DataFrame): A merged DataFrame with duplicate 'Image Link' rows removed.
        """
        combined_df = pd.merge(description_df, image_df, on=merge_col, how='inner')
        combined_df = combined_df.drop_duplicates(subset=['Image Link'])
        return combined_df

    def merge_export_to_csv(self, image_df: pd.DataFrame, description_df: pd.DataFrame, merge_col: str):
        """
        Merges two DataFrames, re-indexes the ID column, and exports to CSV. 
        
        This method performs a merge of two DataFrames and cleans the resulting 
        dataset by resetting the 'id' column to a sequential range starting from 1. 
        It also assigns a specific store name provided by the user to a new column before 
        writing the data to a CSV file.

        Args:
            image_df (pd.DataFrame): The primary DataFrame to merge.
            description_df (pd.DataFrame): The secondary DataFrame to merge.
            merge_col (str): The column of the DataFrame to merge on.  
        Returns:
            df (pd.DataFrame): A DataFrame with store name column added and id column reset to start from 1.
        """
        df = self.merge_drop_duplicates(image_df, description_df, merge_col)

        if "Id_y" in df.columns:
            df = df.drop(columns=['Id_y']) 
            df = df.rename(columns={'Id_x': 'Id'})

        df['Id'] = range(1, len(df) + 1)
        df = df.insert(1, 'Store', self.store_name)
        df = df.to_csv(f"./data/raw/{self.store_name}_items.csv", index=False)
        print("Written to csv.")

        return df

