from lxml import etree
import csv
import re
import os
import argparse
from datetime import datetime
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np

def parseXML(xmlfile, since_date):
    # for each document cat_count will contain a dictionary of the count of each ticket category
    cat_count = {}
    file = open(xmlfile, "r")
    contents = file.read()
    soup = BeautifulSoup(contents, 'xml')
    elements = soup.find_all('item')
    clean = re.compile('<.*?>') # xml characters to clean from the text
    for item in elements:
        updated_last = item.find('updated')
        if(updated_last is None):
            print('skipped')
            continue
        updated_last_date = datetime.strptime(updated_last.text, '%a, %d %b %Y %H:%M:%S %z').replace(tzinfo=None)
        if updated_last_date <= since_date:
            continue
        ticket_id = item.find('key')
        items = []
        tick = 'parsed_tickets/' + ticket_id.text + '.csv'
        # dictionary to prompts and responses
        prompt_response = {}
        initial_question = re.sub(clean, '', str(item.find('description').text))
        prompt_response['prompt'] = initial_question
        # if we want cat_count to be per ticket add here
        cat = item.find("customfield", id="customfield_11300").find("customfieldvalue").text.strip()

        if (cat in cat_count):
            cat_count[cat] += 1
        else:
            cat_count[cat] = 1


        i = 0
        comments = item.find_all('comment')
        if not comments:
            # these tickets only have descriptions
            prompt_response['response'] = ''
            items.append(prompt_response)
            prompt_response = {}
        for comment in comments:
            reply = re.sub(clean, '', str(comment.text))
            if (i%2 == 0):
                prompt_response['response'] = reply
                items.append(prompt_response)
                prompt_response = {}
            else:
                prompt_response['prompt'] = reply
           
            i += 1

        # specifying the fields for csv file 
        fields = ['prompt', 'response'] 

        # writing to csv file 
        with open(tick, 'w') as csvfile: 

            # creating a csv dict writer object from datetime import time
            writer = csv.DictWriter(csvfile, fieldnames = fields) 
            # writing headers (field names) 
            writer.writeheader() 
            # writing data rows 
            writer.writerows(items)
       
    
    return cat_count
         
def mkdate(datestr):
    try:
        return datetime.strptime(datestr, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(datestr + ' is not a proper date string')

def mk_nested_pie_chart(system_data):
    # make a graph using system_counts

    # filter the data to only include counts from specific systems, otherwise the graph is too busy
    data = {k: system_data[k] for k in ['summit', 'frontier', 'andes', 'quantum', 'crusher']}

    size = 0.3
    # see https://matplotlib.org/stable/gallery/pie_and_polar_charts/nested_pie.html for example of simple nested chart code
    # Process data to include only top 3 subcategories + "other" (other is all other categories combined)
    processed_data = {}
    for category, subcats in data.items():
        sorted_subcats = sorted(subcats.items(), key=lambda item: item[1], reverse=True)
        if len(sorted_subcats) > 3:
            top_subcats = sorted_subcats[:3]
            other_value = sum(value for name, value in sorted_subcats[3:])
            top_subcats.append(('Other', other_value))
        else:
            top_subcats = sorted_subcats
        processed_data[category] = dict(top_subcats)

    flat_vals = []
    category_bounds = [0]
    category_labels = []
    subcategory_labels = []
    total_percentages = []  # Percent of the total
    slice_percentages = []  # Percent of the slice
    category_percentages = []

    total = sum(sum(sub.values()) for sub in processed_data.values())

    for category, subcats in processed_data.items():
        category_sum = sum(subcats.values())
        flat_vals.extend(subcats.values())
        category_bounds.append(len(flat_vals))
        category_labels.append(category)
        subcategory_labels.extend(subcats.keys())
        total_percentages.extend([(value / total) * 100 for value in subcats.values()])
        slice_percentages.extend([(value / category_sum) * 100 for value in subcats.values()])
        category_percentages.append((category_sum / total) * 100)

    flat_vals = np.array(flat_vals)
    valsnorm = flat_vals/np.sum(flat_vals)*2*np.pi
    valsleft = np.cumsum(np.append(0, valsnorm[:-1]))

    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

    cmap = plt.colormaps["tab20c"]
    outer_colors = cmap(np.arange(len(processed_data))*4)
    inner_colors = cmap(np.arange(len(flat_vals)))

    # Outer categories
    for i, bound in enumerate(category_bounds[:-1]):
        start_angle = valsleft[bound]
        end_angle = valsleft[category_bounds[i+1]-1] + valsnorm[category_bounds[i+1]-1]
        ax.bar(x=start_angle, width=end_angle - start_angle, bottom=1-size, height=size, color=outer_colors[i], edgecolor='w', linewidth=1, align="edge")
        label_angle = np.degrees((start_angle + end_angle) / 2)
        ax.text(np.radians(label_angle), size + 0.55, f"{category_labels[i]}: {category_percentages[i]:.1f}%", ha='center', va='center', rotation=0, color='white')

    # Inner subcategories
    bars = ax.bar(x=valsleft, width=valsnorm, bottom=1-2*size, height=size, color=inner_colors, edgecolor='w', linewidth=1, align="edge")

    # Legend with detailed percentages
    legend_labels = [f"{label}: {slice_pct:.1f}% ({total_pct:.1f}% of total)" for label, slice_pct, total_pct in zip(subcategory_labels, slice_percentages, total_percentages)]
    legend = ax.legend(bars, legend_labels, loc="upper left", bbox_to_anchor=(1.05, 1), title="Ticket Types")

    ax.set(title="Jira Tickets by Selected Systems")
    ax.set_axis_off()

    plt.show()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('since_date',type=mkdate)
    args=parser.parse_args()
    directory = 'jiratickets'
    system_counts = {}
    
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            counts = parseXML(f, args.since_date)
            system = filename.split('_', 1)[0]
            if system in system_counts:
                for key, value in counts.items():
                    system_counts[system][key] = system_counts[system].get(key, 0) + value
            else:
                system_counts[system] = counts

    mk_nested_pie_chart(system_counts)

    


if __name__ == "__main__":

    main()
