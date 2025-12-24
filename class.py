# >>what to find
# 1.total pass by class
# 2.total pass/fail by class by subject
# 3.highest per subject per class
# 4.highest across all the exams

# dataset preview
# >classes = A and B
# >subjects= math,physics,chemistry,biology,social
# ten students in each class
# each subject >> total=100
# if marks<30 --> fail

# english logic

# 1.total pass by class
# 

# 2.total pass/fail by class by subject
# >> graph 1 >class A > x-axis--> subjects,y-axis--> pass count
# >> graph 2 >class b > x-axis--> subjects,y-axis--> pass count
# >> pass count of students for each subject

# 3.highest per subject per class
# >> graph 1 >class A > x-axis--> subjects,y-axis--> highest score
# >> graph 2 >class b > x-axis--> subjects,y-axis--> highest score

# 4.highest across all the exams
# >> graph > x-axis --> class name,y-axis-->score
# >> 

##code
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("class.csv")
#subject=["Math","Physics","Chemistry","Biology","Social"]
#class,student,exam date,subject, total marks,scored,status
pass_mark=30
passed_student=df[df["scored"]>=pass_mark]
#print(passed_student)

# 2.total pass/fail by class by subject
result=passed_student.groupby(["class","subject"]).size().reset_index(name="pass_count")
#print(result)

df_A=result[result["class"]=='A']
#print(df_A)

df_A.plot(x="subject",y="pass_count",kind="bar",figsize=(10,5))
plt.xlabel("Subject")
plt.ylabel("pass count")
plt.title("subject Vs pass count for class A")
plt.grid(axis="y")
plt.tight_layout()
#plt.show()

df_B=result[result["class"]=='B']
#print(df_B)

df_B.plot(x="subject",y="pass_count",kind="bar",figsize=(10,5))
plt.xlabel("Subject")
plt.ylabel("pass count")
plt.title("subject Vs pass count for class B")
plt.grid(axis="y")
plt.tight_layout()
#plt.show()

# 1.total pass by class
res=df[df["status"]=="PASS"].groupby(["class","student"]).size().reset_index(name="status_count")
#print(res)

df_S=res[res["status_count"]>=4].groupby(['class']).size().reset_index(name="passed")
#print(df_S)

df_S.plot(x="class",y="passed",kind="bar",figsize=(10,5))
plt.xlabel("Class")
plt.ylabel("passed")
plt.title("class Vs passed")
plt.grid(axis="y")
plt.tight_layout()
#plt.show()

# 3.highest per subject per class
hm=df.groupby(["class","subject"])["scored"].max().reset_index(name="Max_Marks")
print(hm)
colors=["red","green"]
plt.hist(hm[hm["class"]=='A']["Max_Marks"],bins=5,alpha=0.6,color='red',label='class A')
plt.hist(hm[hm["class"]=='B']["Max_Marks"],bins=5,alpha=0.6,color='green',label='class B')
subjects=hm["subject"].unique()
#hm.plot(x="subject",y="passed",kind="bar",figsize=(10,5))
plt.xlabel("Subject")
plt.ylabel("Max_Marks")
#plt.legend()
plt.title("subject vs max marks for class A and B")
plt.grid(axis="y")
plt.tight_layout()
#plt.show()


# 4.highest across all the exams
he=df.groupby(["subject"])["scored"].max().reset_index(name="Max_Marks")
print(he)

he.plot(x="subject",y="Max_Marks",kind="bar",figsize=(10,5))
plt.xlabel("subjects")
plt.ylabel("Max Marks")
plt.title("subjects vs max marks")
plt.grid(axis="y")
plt.tight_layout()
plt.show()



