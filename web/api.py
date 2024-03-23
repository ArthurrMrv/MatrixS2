import sys
import json
import ast

data_to_pass_back = "send this node process"

input = ast.literal_eval(sys.argv[1])

with open("data.json", "w") as f:
    json.dump(input, f)

with open("data.txt", "w") as f:
    f.write(str(input))
    f.write(input["message"])


sys.stdout.flush()