import jsonlines
import docutils
from docutils.parsers import rst
from docutils.io import StringInput
from docutils.core import publish_doctree, publish_from_doctree
import os, re, glob

# patterns to fix 
pattern1_to_replace = r':ref:`([^`]+)`'
pattern2_to_replace = r':doc:`([^`]+) <([^`]+)>`'
pattern3_to_replace = r'\\ \|R\|'
patterns_to_remove = [r':linenos:', r'tabbed::', r'glossary::', r':term:', r'dropdown::', r':caption:']
# manual fixed unmatched :doc:



def rst_to_json(input_rst):
    with open(input_rst, 'r') as rst_file:
        rst_text = rst_file.read()

    # fix rst file
    updated_content = re.sub(pattern1_to_replace, r'`\1 <https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#\1>`', rst_text)
    updated_content = re.sub(pattern2_to_replace, r'`\1 <https://docs.olcf.ornl.gov\2.html>`', updated_content)
    updated_content = re.sub(pattern3_to_replace, '     ', updated_content)
     
    for pattern in patterns_to_remove:
        updated_content = re.sub(pattern, '', updated_content)

    #document = publish_doctree(updated_content)
    document = publish_doctree(updated_content, settings_overrides={'report_level':5})

    # extract tables  
    tables = []; counter = 1
    for table_node in document.traverse(docutils.nodes.table):
        table_data = []
        for row in table_node.traverse(docutils.nodes.row):
            row_data = []
            for cell in row.traverse(docutils.nodes.entry):
                cell_data = ""
                for paragraph in cell.traverse(docutils.nodes.paragraph):
                    cell_data = paragraph.astext()
                for reference in cell.traverse(docutils.nodes.reference):
                    if reference.get("refuri"):
                        cell_data = cell_data + " " + reference.get("refuri")
                row_data.append(cell_data)
            table_data.append(row_data)
        placeholder = f"Table-{counter}-to-be-replaced"
        counter += 1
        table_node.replace_self(docutils.nodes.paragraph(text=placeholder))
        tables.append(table_data)

    basename = os.path.splitext(os.path.basename(input_rst))[0]
    table_file = basename + "_tables.txt"
    with open(table_file, 'w') as output_file:
        for i, table in enumerate(tables):
            output_file.write(f"Table {i + 1}:\n")
            for row in table:
                output_file.write("|".join(row) + "\n")
            output_file.write("\n")

    output_json = basename + ".jsonl"
    with jsonlines.open(output_json, 'w') as writer:
        for node in document.traverse():
            if node.__class__.__name__ == 'document':
                writer.write({'doc': basename, 'text': node.astext()})

if __name__ == "__main__":
    input_rst_files = glob.glob("../../data/raw/olcf-user-doc-9-23/*.rst")
    for rst_file in input_rst_files:
        rst_to_json(rst_file)
