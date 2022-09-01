from xml.etree import ElementTree as ET

class ElogInterface:

    def __init__(self, setup):
        self.exp = setup.exp  # experiment name, str
        self.instrument = setup.exp[:3]
        self.summary_dir = f'/cds/data/psdm/{self.instrument}/{self.exp}/stats/summary/'
        self.root_dir = setup.root_dir

    def update_summary(self):
        run_list = get_processed_run_list()

    def get_processed_run_list(self):
        file_list = glob.glob(f'{self.root_dir}/powder/figs/stats*.png')
        run_list = []
        for file in file_list:
            run_list.append(file.split('/')[-1].split('_')[1].split('.')[0])
        return run_list



class HtmlInterface:

    def __init__(self, html_path):
        self.html_path = html_path
        self.html = ET.Element('html')
        self.html.append(ET.Element('head'))
        self.html_body = ET.Element('body')
        self.html.append(self.html_body)

    def add_img(self, image_path=None):
        if image_path is not None:
            div = ET.Element('div', attrib={'class': 'foo'})
            div.append(ET.Element('img', attrib={'src': f'{image_path}'}))
            self.body.append(div)

    def write(self):
        with open(self.html_path, 'w') as f:
            ET.ElementTree(self.html).write(f, encoding='unicode', method='html')