import pandas as pd
import matplotlib.pyplot as plt

 

# ============================================================
# 1. Load data
# ============================================================
FILE_NAME = "class1.csv"   # CSV should have the correct columns

 

df = pd.read_csv(FILE_NAME)

 

# Convert types
df["ExamDate"] = pd.to_datetime(df["ExamDate"], errors="coerce")
df["TotalMarks"] = pd.to_numeric(df["TotalMarks"], errors="coerce")
df["Scored"] = pd.to_numeric(df["Scored"], errors="coerce")

 

print("=== Raw Data (first few rows) ===")
print(df.head())
print()

 

# ============================================================
# 2. Define colour maps
# ============================================================
# You can adjust these as you like
class_colors = {
    "12A": "#1f77b4",   # blue
    "12B": "#ff7f0e",   # orange
    "12C": "#2ca02c",   # green
    "12D": "#d62728",   # red
}

 

subject_colors = {
    "Maths": "#9467bd",     # purple
    "Physics": "#8c564b",   # brown
    "Chemistry": "#e377c2", # pink
    "Biology": "#7f7f7f",   # grey
    "English": "#17becf",   # teal
}

 

status_colors = {
    "Pass": "#2ca02c",   # green
    "Fail": "#d62728",   # red
}

 

# ============================================================
# 3. 1) Total pass by class
# ============================================================
pass_mask = df["Status"].str.lower() == "pass"
pass_by_class = (
    df[pass_mask]
    .groupby("Class")["Student"]
    .count()
    .sort_index()
)

 

print("=== 1) Total Pass by Class ===")
print(pass_by_class)
print()

 

# -------- Plot: Total pass by class (multicolour by class) ----
plt.figure(figsize=(8, 5))

 

# Get colour per class, fallback to a default if not in map
bar_colors = [class_colors.get(c, "#333333") for c in pass_by_class.index]

 

plt.bar(pass_by_class.index, pass_by_class.values, color=bar_colors)
plt.title("Total Pass by Class")
plt.xlabel("Class")
plt.ylabel("Number of Students Passed")
plt.grid(axis="y")

 

for x, y in zip(pass_by_class.index, pass_by_class.values):
    plt.text(x, y + 0.1, str(y), ha="center", va="bottom")

 

plt.tight_layout()
plt.show()

 

# ============================================================
# 4. 2) Total pass/fail by class by subject
# ============================================================
pass_fail_by_class_subject = (
    df.groupby(["Class", "Subject", "Status"])
      .size()
      .unstack(fill_value=0)       # Columns: Fail / Pass
)

 

print("=== 2) Total Pass/Fail by Class by Subject ===")
print(pass_fail_by_class_subject)
print()

 

# Prepare for stacked bar plot
plot_df = pass_fail_by_class_subject.reset_index()
plot_df["Class_Subject"] = plot_df["Class"].astype(str) + " - " + plot_df["Subject"].astype(str)
plot_df = plot_df.set_index("Class_Subject")

 

# Ensure columns exist even if there are no fails or no passes somewhere
if "Pass" not in plot_df.columns:
    plot_df["Pass"] = 0
if "Fail" not in plot_df.columns:
    plot_df["Fail"] = 0

 

plt.figure(figsize=(12, 6))
x_labels = plot_df.index

 

plt.bar(
    x_labels,
    plot_df["Pass"],
    color=status_colors.get("Pass", "green"),
    label="Pass"
)
plt.bar(
    x_labels,
    plot_df["Fail"],
    bottom=plot_df["Pass"],
    color=status_colors.get("Fail", "red"),
    label="Fail"
)

 

plt.title("Pass/Fail by Class and Subject")
plt.xlabel("Class - Subject")
plt.ylabel("Number of Students")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")
plt.legend()
plt.tight_layout()
plt.show()

 

# ============================================================
# 5. 3) Highest per subject per class
# ============================================================
# Get the index of the max 'Scored' per (Class, Subject)
idx = df.groupby(["Class", "Subject"])["Scored"].idxmax()
top_per_subject_class = df.loc[idx, ["Class", "Subject", "Student", "Scored"]]
top_per_subject_class = top_per_subject_class.sort_values(["Class", "Subject"])

 

print("=== 3) Highest per Subject per Class ===")
print(top_per_subject_class)
print()

 

# -------- Plot: Highest per subject per class (colour by subject) ----
plt.figure(figsize=(12, 6))

 

labels = top_per_subject_class["Class"] + " - " + top_per_subject_class["Subject"]
scores = top_per_subject_class["Scored"]

 

bar_colors = [
    subject_colors.get(subj, "#333333") 
    for subj in top_per_subject_class["Subject"]
]

 

plt.bar(labels, scores, color=bar_colors)
plt.title("Highest Score per Subject per Class")
plt.xlabel("Class - Subject")
plt.ylabel("Top Score")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")

 

for x, y, name in zip(labels, scores, top_per_subject_class["Student"]):
    plt.text(x, y + 0.3, f"{name} ({y})", ha="center", va="bottom", fontsize=8)

 

plt.tight_layout()
plt.show()

 

# ============================================================
# 6. 4) Highest across all the exams
# ============================================================
max_idx = df["Scored"].idxmax()
top_overall = df.loc[max_idx]

 

print("=== 4) Highest Across All Exams ===")
print(top_overall)
print()

 

# -------- Plot: Highest overall (colour by status) ----
plt.figure(figsize=(6, 4))

 

overall_color = status_colors.get(top_overall["Status"], "#333333")
label = f"{top_overall['Student']} ({top_overall['Class']} - {top_overall['Subject']})"

 

plt.bar([label], [top_overall["Scored"]], color=[overall_color])
plt.title("Highest Score Across All Exams")
plt.xlabel("Student (Class - Subject)")
plt.ylabel("Score")
plt.ylim(0, max(df["Scored"]) * 1.1)

 

for x, y in zip([label], [top_overall["Scored"]]):
    plt.text(x, y + 0.3, str(y), ha="center", va="bottom")

 

plt.tight_layout()
plt.show()
 

