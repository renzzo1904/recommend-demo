""" 

Helper Functions --------------------------------

"""

################################################################################################


def generate_inventory(your_df, tag_column: str = "tags", separator: str = " "):
    """
    Helper function that creates a master inventory for all the products in historical DataFrame.

    Args:
        df (DataFrame): The historical DataFrame containing product information.
        tag_column (str,optional) -- default:'tags' -- : The name of the column containing product tags.
        separator (str, optional) -- default:' " " ' -- : The separator used to split tags. Default is a space.

    Returns:
        dict: A dictionary representing the product inventory with tag counts.
    """

    inv = {}

    for tags in your_df[tag_column]:
        tag_list = tags.split(separator)
        for tag in tag_list:
            tag = tag.strip()  # Remove leading/trailing spaces
            if tag:
                inv[tag] = inv.get(tag, 0) + 1  # Increment tag count

    return inv


################################################################################################


def create_profiles(
    your_df, user_idex_column: str = "user_id", tag_column: str = "tags"
):
    """
    Helper function that creates a unique row for just one user, including all purchases or likes in all rows of historical purchases.

    Args:
        your_df (DataFrame): The historical DataFrame containing user information.
        user_idex_column (str): The column name representing the user ID. Default is 'user_id'.
        tag_column (str,optional) -- default:'tags' -- : The name of the column containing product tags.

    Returns:
        DataFrame: A DataFrame containing user profile information.
    """
    # Group by user_id and concatenate the products into a single row
    user_profiles = (
        your_df.groupby(user_idex_column)[tag_column]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )
    user_profiles.rename(
        columns={user_idex_column: "user_id", tag_column: "likes"}, inplace=True
    )

    return user_profiles


################################################################################################


# def search_similar_users(your_df, user_likes:list ,user_idx_column:str):

#     """
#     Helper function that returns the a Series containing a similarity rank for user similarity.

#     Args:
#         df (DataFrame): The historical DataFrame containing user information.
#         user_likes (list): Likes of the user that is going to be used as central criteria.
#         user_idx_column (str): Name of the column in df that contains the historical user likes.

#     Returns:
#         series : A Series of Unique User indices that display similarity """
