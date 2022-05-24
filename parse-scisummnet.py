###############################
# ARCHIVE ONLY - WILL NOT RUN #
###############################

# Imports
import os
import xmltodict

# Get a list of IDs
with open("scisummnet-parsed/all-ids.txt", "r") as f:
    doc_ids = f.read().splitlines()

# Iterate over IDs
for doc_id in doc_ids:

    # Print document ID to start
    print(f"Starting to parse {doc_id}...", end = "")

    # Load in the XML as a dictionary
    with open(f"scisummnet/top1000_complete/{doc_id}/Documents_xml/{doc_id}.xml", "r") as f:
        doc = xmltodict.parse(f.read())
        
    # If there's no abstract, this isn't a valid document we can use
    try:
        if doc["PAPER"]["ABSTRACT"] == None:
            print("abstract not found. Skipping.")
            continue
    except KeyError:
        print("abstract not found. Skipping.")
        continue
        
    # If there's only one section, this isn't a valid document we can use
    try:
        if type(doc["PAPER"]["SECTION"]) == dict:
            print("only one section found. Skipping.")
            continue
    except KeyError:
        print("sections not found. Skipping.")
        continue

    # Get the abstract
    if type(doc["PAPER"]["ABSTRACT"]["S"]) == dict:
        abstract = doc["PAPER"]["ABSTRACT"]["S"]["#text"]
    else:
        abstract = " ".join(s["#text"] for s in doc["PAPER"]["ABSTRACT"]["S"])

    # Make a line for each section
    text = []
    for section in doc["PAPER"]["SECTION"]:
        title = [section["@title"] + "."]
        try:
            if type(section["S"]) == dict:
                sentences = [section["S"]["#text"]]
            else:
                sentences = [s["#text"] for s in section["S"]]
        except KeyError:
            sentences = []
        text.append(" ".join(title + sentences))

    # Create a directory for the parsed text if it doesn't exist
    output_dir = f"scisummnet-parsed/{doc_id}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the abstract
    with open(f"{output_dir}/abstract.txt", "w") as f:
        f.write(abstract)

    # Write the full text
    with open(f"{output_dir}/full.txt", "w") as f:
        f.write("\n".join(text))

    # Print end of line
    print("finished.")