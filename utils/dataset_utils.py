import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split


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

#asthe_gender = asthe["gender"].value_counts(normalize=True)*100
#print("Aesthetics gender counts:", asthe_gender)

bins = [18,22,26,30,35,41]
labels = ["18-21","22-25","26-29","30-34","35-40"]
lapis["age_groups"] = pd.cut(lapis["age"],bins=bins, labels=labels,right=False)
lapis_age_grouped = lapis["age_groups"].value_counts(normalize=True)*100

l_age = lapis["age"].value_counts(normalize=True)*100
print("Lapis ages:",lapis_age_grouped)
#asthe_age = asthe["age"].value_counts(normalize=True)*100
#print("Aesthetics ages:", asthe_age)


#l_aesth_score = lapis["rating"].value_counts()
#print("Lapis aesthetic scores:", l_aesth_score)
#asthe_score = para["aestheticScore"].value_counts()
#print("Aesthetic scores:", asthe_score)

#l_edu = lapis["demo_edu"].value_counts(dropna=False)
#print("LAPIS edu:", l_edu)
#asthe_edu = asthe["EducationalLevel"].value_counts(dropna=False)
#print("Aesthetics edu:", asthe_edu)

################ MALE ANNOTATORS ###############
#Check if some instances have less men than women annotations
#participant_id , gender -> number of ratings
lapis_participants = lapis.groupby(["participant_id","demo_gender"])["image_id"].count().sort_values()
print(lapis_participants)

lapin_men = lapis[lapis["demo_gender"]=="male"].groupby(["participant_id","demo_gender"])["image_id"].count().sort_values()

print("Men participants rating:", lapin_men)


### Cut-of the 10% of participants with low rates ###

new_lapis = lapis.copy()

male_counts = new_lapis[new_lapis["demo_gender"]=="male"].groupby("participant_id")["image_id"].count()

cutoff = male_counts.quantile(0.20)
remove_male_ids = male_counts[male_counts <= cutoff].index

#keep eveything that is not male and low-rate participant
lapis_filtered = new_lapis[~((new_lapis["demo_gender"]=="male")&(new_lapis["participant_id"].isin(remove_male_ids)))]

print("Male cutoff (ratings):", cutoff)
print("Removed male participants:", len(remove_male_ids))

lapis_demo_gen_fil = lapis_filtered["demo_gender"].value_counts(normalize=True)*100
print("Lapis new percentages:", lapis_demo_gen_fil)


## Measure 1: Find annotator percentage per instance 

lapis_gens = lapis_filtered[lapis_filtered["demo_gender"].isin(["male", "female"])].copy()
lapis_per_image_gender = lapis_gens.groupby(["image_id", "demo_gender"])["participant_id"].nunique().unstack(fill_value=0)

lapis_per_image_gender["total_annotators"] = lapis_per_image_gender.sum(axis=1)

lapis_per_image_gender["female_pct"] = lapis_per_image_gender["female"]/lapis_per_image_gender["total_annotators"]
lapis_per_image_gender["male_pct"] = lapis_per_image_gender["male"]/lapis_per_image_gender["total_annotators"]


print("Lapis percantages of female image annotators :",lapis_per_image_gender["female_pct"] )

print("Lapis percentages of male image annotators:",lapis_per_image_gender["male_pct"] )

## Measure 2: Find the maximun rating per image for males

lapis_image_score_max = lapis_filtered[lapis_filtered["demo_gender"]=="male"].groupby("image_id")["rating"].max()

lapis_im_male_score_sorted = lapis_image_score_max.sort_values(ascending=False)
print(lapis_im_male_score_sorted.head(20))

lapis_image_level = lapis_per_image_gender.copy()
lapis_image_level["male_max_rating"] = lapis_image_score_max
lapis_image_level = lapis_image_level.sort_values("male_max_rating",ascending=False)

print(lapis_image_level.head(20))

## keep the 20% of high male rates as possitive and the bottom 80% as negative

cutoff = lapis_image_level["male_max_rating"].quantile(0.80)
lapis_image_level["label_top20_male_max"] = (lapis_image_level["male_max_rating"]>= cutoff).astype(int)

print(lapis_image_level["label_top20_male_max"].value_counts(normalize=True))
print(lapis_filtered["rating"].describe())
