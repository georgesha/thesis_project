import xml.etree.ElementTree
import csv
import glob
import os

xml_path = ""
csv_path = ""
xml_files = glob.glob(os.path.join(xml_path, '*xml'))
csv_files = []

for xml_file in xml_files:
    parser = xml.etree.ElementTree.XMLParser(encoding="utf-8")
    root = ""
    try:
        root = xml.etree.ElementTree.parse(xml_file, parser=parser).getroot()
    except Exception as e:
        print(xml_file)
        continue
    csv_file = csv_path + xml_file.split("\\")[-1].split(".")[0] + ".csv"
    csv_files.append(csv_file)
    file = open(csv_file, "w", newline="")
    writer = csv.writer(file, delimiter=",", lineterminator="\n")
    writer.writerow(["Assession", "Label", "Diseases - term:observation:label"])
    for document in root.iter('document'):
        if len(document) < 2:
            continue
        assession = document[0].text
        passages = iter(document[1:])
        labels = []
        diseases = ""
        term = ""
        observation = ""
        for passage in passages:
            # for infon in passage.iter('infon'):
            #     if infon.attrib["key"] == "title":
            #         title = infon.text
            #         if title == "Clinical history":
            #             next(passages)
            #             continue
            for annotation in passage.iter('annotation'):
                cur_negation = False
                cur_uncertainty = False
                for infon in annotation.iter('infon'):
                    if infon.attrib["key"] == "term":
                        term = infon.text
                    if infon.attrib["key"] == "observation":
                        observation = infon.text
                    if infon.attrib["key"] == "negation" and infon.text == "True":
                        cur_negation = True
                    if infon.attrib["key"] == "uncertainty" and infon.text == "True":
                        cur_uncertainty = True
                if observation == "No Finding":
                    continue
                diseases = diseases + term + ":" + observation + ":"
                if cur_negation:
                    diseases = diseases + "0;"
                    labels.append("0")
                elif cur_uncertainty:
                    diseases = diseases + "-1;"
                    labels.append("-1")
                else:
                    diseases = diseases + "1;"
                    labels.append("1")
        if "1" in labels:
            writer.writerow([assession, "1", diseases])
        elif "-1" in labels:
            writer.writerow([assession, "-1", diseases])
        else:
            writer.writerow([assession, "0", diseases])

# combine csv
target_file =  open(csv_path + "full_report.csv", "w", newline="")
writer = csv.writer(target_file, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')
for csv_file in csv_files:
    file = open(csv_file, "r")
    reader = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONE)
    next(reader)
    for line in reader:
        writer.writerow(line)
