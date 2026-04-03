import docx
doc = docx.Document(r'd:\ML\RL project\Pipeline_Debug_Env_Architecture.docx')
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join([p.text for p in doc.paragraphs]))
