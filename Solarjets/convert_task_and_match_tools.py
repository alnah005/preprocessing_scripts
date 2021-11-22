import pandas as pd
from collections import defaultdict

proposed_file = "point_extractor_by_frame_beta2_proposed.csv"
file_extract: pd.DataFrame = pd.read_csv(proposed_file)

task_to_overide = "T1"
task_to_use_for_override = "T5"
frame_to_overlap = "frame0"

assert "task" in file_extract.columns
assert len([i for i in file_extract.columns if "data.frame" in i])> 0 and len([i for i in file_extract.keys() if "data."+frame_to_overlap in i])> 0

available_frame_columns = [i for i in file_extract.columns if "data.frame" in i]
tools = list(set([i.split('tool')[1].split('_')[0] for i in available_frame_columns]))
unrelated_frame_columns = [i for i in file_extract.columns if "data.frame" not in i]

## need to check that for each extract, only one label was created


for index, row in file_extract.iterrows():
    tool_per_frame = {t:None for t in tools}
    null_cols = pd.isna(row)
    for col in available_frame_columns:
        if not(null_cols[col]):
            frame_number = col.split("data.frame")[1].split('.')[0]
            tool = col.split('tool')[1].split('_')[0]
            assert  tool_per_frame[tool] is None or frame_number == tool_per_frame[tool]
            tool_per_frame[tool] = frame_number

### if above passes, that means there is only one label per extract row
result = defaultdict(list)
for index, row in file_extract.iterrows():
    tool_per_frame = {t:None for t in tools}
    null_cols = pd.isna(row)
    modified = False
    for col in available_frame_columns:
        if not(null_cols[col]):
            frame_number = col.split("data.frame")[1].split('.')[0]
            tool = col.split('tool')[1].split('_')[0]
            new_col = "data."+frame_to_overlap+"."+task_to_use_for_override+"_tool"+tool+"_"+col.split('tool')[1].split('_')[1]
            result[new_col].append(row[col])
            modified = True
    if modified:
        for col in unrelated_frame_columns:
            if col == "task":
                result[col].append(task_to_use_for_override)
            else:
                result[col].append(row[col])

length = [len(i) for i in result.values()]
for i in range(len(length)-1):
    assert length[i] == length[i+1]

new_pd = pd.DataFrame.from_dict(result)
cols = unrelated_frame_columns+[i for i in list(new_pd.keys()) if i not in unrelated_frame_columns]
rearranged_pd = new_pd[cols]
rearranged_pd.to_csv(proposed_file,index=False)
