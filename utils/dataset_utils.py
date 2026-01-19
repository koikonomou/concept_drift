import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
#################### PATH ##########################
data_dir = Path(os.getenv("DATASETS"))
lapis_path = data_dir/"LAPIS"/"LAPIS github"/"annotation"/"LAPIS_PIAA.csv"
aesthetics_user = data_dir/"PARA"/"annotation"/"PARA-UserInfo.csv"
aesthetics_images = data_dir/"PARA"/"annotation"/"PARA-Images.csv"


######### Load csv############
lapis = pd.read_csv(lapis_path)
#print(lapis.head())
print("LAPIS columns:", lapis.columns)
asthe = pd.read_csv(aesthetics_user)
print("Aesthetics users columns:", asthe.columns )
para = pd.read_csv(aesthetics_images)
print("Aesthetics images columns:", para.columns)
#l_nationality = lapis["nationality"].value_counts(normalize=True)*100
#print("Lapis nationality counts:", l_nationality)

########### Labels #############
l_gender = lapis["demo_gender"].value_counts(normalize=True)*100
print("Lapis gender counts:", l_gender)

asthe_gender = asthe["gender"].value_counts(normalize=True)*100
print("Aesthetics gender counts:", asthe_gender)

bins = [18,22,26,30,35,41]
labels = ["18-21","22-25","26-29","30-34","35-40"]
lapis["age_groups"] = pd.cut(lapis["age"],bins=bins, labels=labels,right=False)
lapis_age_grouped = lapis["age_groups"].value_counts(normalize=True)*100

l_age = lapis["age"].value_counts(normalize=True)*100
print("Lapis ages:",lapis_age_grouped)
asthe_age = asthe["age"].value_counts(normalize=True)*100
print("Aesthetics ages:", asthe_age)


l_aesth_score = lapis["rating"].value_counts()
print("Lapis aesthetic scores:", l_aesth_score)
asthe_score = para["aestheticScore"].value_counts()
print("Aesthetic scores:", asthe_score)

l_edu = lapis["demo_edu"].value_counts(dropna=False)
print("LAPIS edu:", l_edu)
asthe_edu = asthe["EducationalLevel"].value_counts(dropna=False)
print("Aesthetics edu:", asthe_edu)

################ FEMALE ANNOTATORS ###############
#Check if some instances have less men than women annotations

lapis_participants = lapis.groupby["image_id"]
print(lapis_participants)


