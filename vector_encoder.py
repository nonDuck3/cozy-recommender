import torch
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from transformers.image_utils import load_image
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import glob

def encode_images_with_text(local_path: str, scraped_df: pd.DataFrame):
    """
    Encodes images and text from a pandas DataFrame into a combined embedding.

    This function takes a pandas DataFrame containing image and text data, 
    processes them using the SiglipModel, and generates a unified embedding 
    by combining both the image and text embeddings.
    Args:
        local_path (str): Gets the tensors and encoder that was loaded locally.
        scraped_df (pd.DataFrame): A DataFrame with the scraped data containing text and images for encoding
          and generating embeddings with Siglip.
    """

    image_embeddings_list = get_embeddings(local_file_path=local_path, df=scraped_df, col_name="Image Link")
    text_embeddings_list = get_embeddings(local_file_path=local_path, df=scraped_df, col_name="Clothing", encode_images=False)

    scraped_df["image_embeddings"] = image_embeddings_list
    scraped_df["text_embeddings"] = text_embeddings_list

    scraped_df["combined_embeddings"] = scraped_df.apply(lambda row: torch.cat((row["image_embeddings"], row["text_embeddings"])), axis=1)

    store_name = scraped_df.loc[0, 'Store']
    scraped_df.to_pickle(f"./data/embeddings/{store_name}_embeddings.pkl")
    print("Saved in pickle file!")

def get_embeddings(local_file_path: str, df: pd.DataFrame, col_name: str, encode_images: bool = True):

    """
    Generates embeddings for images or text using a local SigLIP model.

    This function loads a SigLIP model and processor from a local directory 
    to encode data from a specific DataFrame column. It handles both visual 
    and textual inputs based on the encode_image flag.

    Args:
        local_file_path (str): The local file system path to the directory containing 
            the SigLIP model weights and configuration.
        df (pd.DataFrame): The pandas DataFrame containing the data to be 
            embedded.
        col_name (str): The name of the column in `df` to process. 
        encode_image (bool): Determines the encoding mode. If True, treats the 
            input as image links; if False, treats the input as text. 
            Defaults to True.

    Returns:
        embeddings_list (list): A list where each element is an embedding (typically a 1D tensor 
            or list of floats) representing the encoded record for each row 
            in the input DataFrame.
    """

    model = AutoModel.from_pretrained(local_file_path, local_files_only=True)

    embeddings_list = []
    for item in tqdm(df[col_name], desc="Encoding & Generating Embeddings..."):
        try:
            if encode_images:
                image = load_image(item)
                processor = get_siglip_model(file_path=local_file_path)
                inputs = processor(images=image, return_tensors="pt")

            else: 
                tokenizer = get_siglip_model(file_path=local_file_path, encode_image=False)
                inputs = tokenizer(item, padding="max_length", return_tensors="pt")

            with torch.no_grad():
                if encode_images:
                    features = model.get_image_features(**inputs)
                else:
                    features = model.get_text_features(**inputs)
                
                tensors = features.pooler_output
                embeddings_list.append(tensors)

        except Exception as e:
            print(f"Error processing {item}: {e}")
            break

    return embeddings_list

def get_siglip_model(file_path: str, encode_image: bool = True):
    """
    Loads a pre-trained model and tokenizer from a specified file path.
    Args:
        file_path (str): The file path to the directory containing the pre-trained model and tokenizer.
        encode_image (bool): Determines the encoding mode. If True, treats the 
            input as image links; if False, treats the input as text. 
            Defaults to True.
    Returns:
        Union[PreTrainedProcessor, PreTrainedTokenizerBase]: 
        A variable containing the loaded model and tokenizer.
    """

    if encode_image:
        return AutoProcessor.from_pretrained(file_path)
    else:
        return AutoTokenizer.from_pretrained(file_path)
    

def compute_cosine_similarity(file_path: str, user_pref: str, file: str, max_price: float = 100000):
    """
    Computes the cosine similarity between a user's preference (text input) and product embeddings 
    in a given DataFrame, filtered by a specified price range.
    
    Args:
        file_path (str): The file path to the directory containing the pre-trained model.
        user_pref (str): A string representing the user's preference, which will be converted into a vector embedding.
        file (str): The file with scraped data containing clothes data, including a "Price" column and 
                           a "combined_embeddings" column with precomputed product embeddings.
        max_price (float, optional): The maximum price for filtering products. Defaults to 100000.
    
    Returns:
        top_k_results_df (pd.DataFrame): A DataFrame containing the top 10 products with the highest cosine similarity 
                      to the user's preference, sorted in descending order of similarity.
    """

    df = pd.read_pickle(file)
    df["Price"] = df["Price"].str.replace("$", "", regex=False).astype(float)
    df = df[df["Price"] <= max_price]

    model = AutoModel.from_pretrained(file_path)
    tokenizer = get_siglip_model(file_path=file_path, encode_image=False)
    inputs = tokenizer(user_pref, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        user_embeddings = model.get_text_features(**inputs)
        user_embeddings_tensor = user_embeddings.pooler_output
    user_embeddings_tensor = user_embeddings_tensor.numpy().reshape(1, -1)

    similarity_score_list = []
    for product_embeddings in df["combined_embeddings"]:
        single_product_embedding = product_embeddings.mean(dim=0)
        pooled_embedding_np = single_product_embedding.numpy().reshape(1, -1)
        similarity_scores = cosine_similarity(pooled_embedding_np, user_embeddings_tensor)
        similarity_score_val = similarity_scores[0,0]
        similarity_score_list.append(similarity_score_val)

    df["cosine_similarity"] = similarity_score_list

    k = 10
    top_k_items = (
        df.sort_values("cosine_similarity", ascending=False)
        .head(k)
    )
    top_k_results_df = top_k_items.drop(columns=["Id", "image_embeddings", "text_embeddings", "combined_embeddings", "cosine_similarity"])
    return top_k_results_df

def combine_all_embeddings(folder: str, file_name: str):
    """
    Concatenate multiple pickle files containing DataFrames from a folder into a single DataFrame.

    This function searches the specified folder for all files with a `.pkl` extension,
    reads each file as a pandas DataFrame, and concatenates them row-wise into one DataFrame.
    The resulting DataFrame has its index reset for a continuous sequence.

    Args:
        folder (str): Path to the folder containing pickle files to be concatenated.
        file_name (str): File path of combined embeddings data.
    """

    pickle_files = glob.glob(folder)

    all_dfs = []
    for file in tqdm(pickle_files, desc="Concatenating data..."):
        df = pd.read_pickle(file)  
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    combined_df.to_pickle((file_name))
    print("Combined files written to pickle.")