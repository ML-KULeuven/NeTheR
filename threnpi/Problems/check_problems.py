import json
# from threnpi_old.Main.problem_generation.generate_problem import clean_query
from threnpi.src.propositional import query_prop
from threnpi.Data.dataloaders.animal_loader import load_small_animal_data

full_data, cat_cols, drop_cols=load_small_animal_data()
print(full_data.shape)
full_data=full_data[cat_cols]
print(full_data.shape)
for it in range(0,10):
    for depth in range(2,6):
        for num in range(1,5):
            with open("Problems/animals/"+str(depth)+"_"+str(num)+"_"+str(it)+"_small.json") as f:
                problem=json.load(f)
            print(depth, num, it)
            input=problem["input_query"]
            print(input)
            print("Input:", query_prop(input, "P", full_data).shape)
            dt=problem["DT"]
            print(dt)
            print("DT: ", query_prop(dt, "P", full_data).shape)
            rules=problem["Rules"]
            print(rules)
            print("Rules: ", query_prop(rules, "P", full_data).shape)

# with open("Problems/animals/"+str(3)+"_"+str(2)+"_"+str(2)+"_small.json") as f:
#     problem=json.load(f)
# print(problem["Rules_full"])
# print(rules2prop(problem["Rules_full"]))
# print(clean_query(rules2prop(problem["Rules_full"])))