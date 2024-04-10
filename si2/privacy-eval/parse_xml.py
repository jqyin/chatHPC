import xml.etree.ElementTree as ET
from lxml import html
import os
import csv
import re
import pandas as pd

def parse_user_info(filename):
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
    except:
        tree = html.parse(filename)
        root = tree.getroot()

    info = []
    for item in root.iter('item'):
        for reporter in item.findall('reporter'):
            username = reporter.attrib['username']
            name = reporter.text.replace('[X]', '').strip()

        for title in item.findall('title'):
            proj_id_matches_title = set([ match.lower() for match in re.findall(r'\b[A-Za-z]{3}[0-9]{3}\b', title.text)])

        for description in item.findall('description'):
            if description.text != None:
                proj_id_matches_description = set([ match.lower() for match in re.findall(r'\b[A-Za-z]{3}[0-9]{3}\b', description.text)])
            else:
                proj_id_matches_description = set()

        union_proj_ids = proj_id_matches_title.union(proj_id_matches_description)
        if len(union_proj_ids) == 0:
            union_proj_ids = {}
        info.append([username, name, union_proj_ids])
    
    return info


def scrape_all_names(root):
    df_info = pd.DataFrame()
    for file in os.listdir(root):
        if file.endswith(".xml") and file[0] != '.': #and file != 'summit_2.xml' and file != 'summit_3.xml' and file != 'summit_4.xml':
            file_info = parse_user_info(os.path.join(root, file))
            df_info = pd.concat([df_info, pd.DataFrame(file_info)])
    return df_info

def write_to_csv(rows, filename, fields):
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)   

        # writing the fields
        csvwriter.writerow(fields)
    
        # writing the data rows
        csvwriter.writerows(rows)

def remove_duplicates(df):

    df_nodup = pd.DataFrame(columns=['username', 'name', 'projects'])
    names_list = list(set(df['name'].to_list()))
    for name in names_list:
        name_rows = df.loc[df['name'] == name]
        username = name_rows['username'].iloc[0]
        project_sets = name_rows['projects'].tolist()
        project_sets[:] = [set() if x == {} else x for x in project_sets]
        df_nodup.loc[len(df_nodup.index)] = [username, name, set.union(*project_sets)]    
    return df_nodup

def remove_names(df):
    df_no_names = pd.DataFrame(columns=['username', 'name', 'projects'])
    
    for index, row in df.iterrows():
        if row['username'] not in row['name'].lower():
            df_no_names.loc[len(df_no_names.index)] = [row['username'], row['name'], row['projects']]

    return df_no_names

def write_names_set(df, outfile):
    names_list = []
    for index, row in df.iterrows():
        name = row['name']
        tmp_df = df.drop([index], axis=0)
        alt_names = tmp_df.sample(n=6)['name'].tolist()
        names_list.append([row['name']] + alt_names)

    names_df = pd.DataFrame(names_list)
    names_df.to_csv(outfile, index=False, header=False)


info_df = scrape_all_names('.')
info_df.columns = ['username', 'name', 'projects']
df_nodups = remove_duplicates(info_df)
df_nodups.to_csv('data_full.csv')

# write all names to csv
write_names_set(df_nodups, 'names_all.csv')

# drop names with no projects
df_nodups_no_emails = df_nodups.drop(df_nodups[df_nodups.name.str.contains('@')].index)
# write names csv
write_names_set(df_nodups_no_emails, 'names_no_emails.csv')

# remove names that are same as usernames
df_nodups_no_emails_no_names = remove_names(df_nodups_no_emails)
write_names_set(df_nodups_no_emails_no_names, 'names_no_emails_no_usernames.csv')

# drop names with no projects
df_nodups_projects = df_nodups.drop(df_nodups[df_nodups.projects == set()].index)
# write names csv
write_names_set(df_nodups_projects, 'names_projects.csv')

# drop names with no projects
df_nodups_projects_no_emails = df_nodups_no_emails.drop(df_nodups_no_emails[df_nodups_no_emails.projects == set()].index)
# write names csv
write_names_set(df_nodups_projects_no_emails, 'names_projects_no_emails.csv')

    
