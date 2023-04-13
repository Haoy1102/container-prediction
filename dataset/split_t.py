import os
import json

directory = "./dev-nodes"
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".json"):
            filepath = subdir + os.sep + file

            # Read the content of the JSON file and format it with indentation
            with open(filepath) as f:
                data = json.load(f)

            with open(filepath, 'w') as f:
                f.write(json.dumps(data, indent=4))
