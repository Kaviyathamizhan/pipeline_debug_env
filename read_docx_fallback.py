import zipfile
import xml.etree.ElementTree as ET

namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

def read_docx(path):
    with zipfile.ZipFile(path) as zf:
        xml_content = zf.read('word/document.xml')
        tree = ET.fromstring(xml_content)
        paragraphs = tree.findall('.//w:p', namespaces)
        
        doc_text = []
        for p in paragraphs:
            texts = [node.text for node in p.findall('.//w:t', namespaces) if node.text]
            if texts:
                doc_text.append(''.join(texts))
            
        return '\n'.join(doc_text)

with open('output_fallback.txt', 'w', encoding='utf-8') as f:
    f.write(read_docx(r'd:\ML\RL project\Pipeline_Debug_Env_Architecture.docx'))
