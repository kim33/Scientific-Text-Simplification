import os
from openai import OpenAI

def gpt_simplify(text) :
    model_type = 'gpt-4o'
    client = OpenAI(api_key = "")
    response = client.chat.completions.create(
        model= model_type,
        messages=[
            {
                "role" : "system",
                "content" : f"You are a language expert whose goal is to simplify the given text easy to read and understand for non-experts in scientific field and general public. While simplifying, detect any complex words or key phrases. Ensure to replace complex phrases to alternatives or add short explanation. Do not change the format of the paragraph or add new line."
            },
            {"role": "user",
             "content" : f"Simplify the following text. Keep simplified results in the same line. The first line is the title of the paper so keep the first line as it is in the result and move on to the next line for simplification. Put the first line(title) and the simplified texts in the same line. :{text}"
            }
        ],
        max_tokens=1500,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def concatenate_summaries(base_folder):
    # Create or open the summary.txt file to write the concatenated text
    count = 0
    with open("simplified_summary_full.txt", "w") as simplified_file :
        with open("original_summary_full.txt", "w") as summary_file :
            sorted_folder = sorted(os.listdir(base_folder))
            # Iterate through all parent folders in the base folder
            for parent_folder in sorted_folder:
                parent_folder_path = os.path.join(base_folder, parent_folder)
                
                # Check if the parent folder is a directory
                if os.path.isdir(parent_folder_path):
                    # Iterate through the subfolders in the parent folder
                    for subfolder_name in os.listdir(parent_folder_path):
                        if 'summary' in subfolder_name :
                            summary_folder_path = os.path.join(parent_folder_path, subfolder_name)
                            if os.path.isdir(summary_folder_path):
                            # Iterate through each text file in the "summary" folder
                                for file_name in os.listdir(summary_folder_path):
                                    file_path = os.path.join(summary_folder_path, file_name)
                                # Check if the file is a .txt file
                                    if file_name.endswith(".txt") and os.path.isfile(file_path) :
                                    # Open and read the content of the text file
                                        with open(file_path, "r") as file:
                                            content = file.read()
                                            summary_file.write(content + "\n\n")
                                            simplified_text = gpt_simplify(content)
                                        # Write the content into summary.txt
                                            simplified_file.write(simplified_text + "\n\n")  # Add newline to separate summaries
                                    else :
                                        return

# Base folder where all parent folders are located
base_folder = "../scisummnet/top1000_complete/"  # Replace with the actual base folder path
concatenate_summaries(base_folder)
