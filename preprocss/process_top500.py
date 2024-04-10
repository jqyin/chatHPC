import xml.etree.ElementTree as ET
import csv

def parse_xml(file_path, output_file):
    tree = ET.parse(file_path)
    root = tree.getroot()

    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["System Name", "Rank", "Manufacturer", "R-Max", "Power", "Installation Site"])

        for site in root.findall('.//{http://www.top500.org/xml/top500/1.0}site'):
            rank = site.find('{http://www.top500.org/xml/top500/1.0}rank').text
            system_name = site.find('{http://www.top500.org/xml/top500/1.0}system-name').text
            manufacturer = site.find('{http://www.top500.org/xml/top500/1.0}manufacturer').text
            r_max = site.find('{http://www.top500.org/xml/top500/1.0}r-max').text
            power = site.find('{http://www.top500.org/xml/top500/1.0}power').text
            installation_site_name = site.find('.//{http://www.top500.org/xml/top500/1.0}installation-site-name').text

            csv_writer.writerow([system_name, rank, manufacturer, r_max, power, installation_site_name])

parse_xml('TOP500_202311_all.xml', 'top500.csv')
