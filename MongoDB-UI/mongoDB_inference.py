import sys
import json
import os
import csv
import requests
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QRadioButton, QCheckBox,
                             QLineEdit, QVBoxLayout, QHBoxLayout, QMessageBox, QGroupBox, QFormLayout,
                             QFileDialog, QTextEdit, QDialog, QScrollArea, QMainWindow,QSpacerItem, QSizePolicy,QComboBox)
from PyQt5.QtCore import Qt
from pymongo import MongoClient
from bson import ObjectId
from PIL import Image
from io import BytesIO
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QPixmap


class ViewImageDialog(QDialog):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.setWindowTitle('Enter Image ID')
        self.setMinimumSize(300, 100)

        # Main layout
        main_layout = QVBoxLayout()

        # Input for Image ID and View button
        input_layout = QHBoxLayout()
        self.id_input = QLineEdit(self)
        self.id_input.setPlaceholderText('Enter Image ID')
        self.id_input.setStyleSheet("font-weight: bold;")
        input_layout.addWidget(self.id_input)

        self.view_button = QPushButton('View Image', self)
        self.view_button.setStyleSheet("font-weight: bold;")
        self.view_button.clicked.connect(self.open_image_display_dialog)
        input_layout.addWidget(self.view_button)

        main_layout.addLayout(input_layout)
        self.setLayout(main_layout)

    def open_image_display_dialog(self):
        image_id = self.id_input.text()
        if not image_id:
            QMessageBox.warning(self, 'Warning', 'Please enter an image ID.')
            return

        try:
            # Search the record using the 'image' field in Image_Descriptions
            image_record = self.db['Image_Descriptions'].find_one({'image': image_id})
            if not image_record:
                QMessageBox.warning(self, 'Error', 'No image found with the given ID.')
                return

            # Open the image display window
            image_display_dialog = ImageDisplayDialog(self.db, image_record)
            image_display_dialog.exec_()
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'An error occurred: {str(e)}')


class ImageDisplayDialog(QDialog):
    def __init__(self, db, image_record):
        super().__init__()
        self.db = db
        self.image_record = image_record
        self.setWindowTitle('View Image')
        self.setMinimumSize(800, 400)
        self.fullscreen = False  # Flag to track fullscreen mode

        # Set general style
        self.setStyleSheet("""
            background-color: #E0FFFF;
            color: black;
            font-family: Arial;
            font-size: 16px;
        """)

        # Add fullscreen button to the title bar
        self.fullscreen_button = QPushButton("Fullscreen", self)
        self.fullscreen_button.setStyleSheet("font-weight: bold;")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        title_bar_layout = QHBoxLayout()
        title_bar_layout.addWidget(self.fullscreen_button)
        title_bar_layout.addStretch()

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(title_bar_layout)

        content_layout = QHBoxLayout()

        # Scroll area for image and Description information
        description_scroll = QScrollArea()
        description_scroll.setWidgetResizable(True)
        self.description_widget = QWidget()
        self.description_layout = QVBoxLayout(self.description_widget)
        description_scroll.setWidget(self.description_widget)
        description_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        description_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content_layout.addWidget(description_scroll, 1)

        # Scroll area for Case and Topic information
        case_topic_scroll = QScrollArea()
        case_topic_scroll.setWidgetResizable(True)
        self.case_topic_widget = QWidget()
        self.case_topic_layout = QVBoxLayout(self.case_topic_widget)
        case_topic_scroll.setWidget(self.case_topic_widget)
        case_topic_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        case_topic_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content_layout.addWidget(case_topic_scroll, 2)

        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

        # Load and display data
        self.load_image_and_info()

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.showNormal()
        else:
            self.showFullScreen()
        self.fullscreen = not self.fullscreen

    def load_image_and_info(self):
        try:
            # Extract image path from record
            image_path = rf"MedPix-2-0/{self.image_record['image']}"

            # Retrieve 'Case' and 'Topic' information from the 'Image_Reports' view
            image_reports_collection = self.db['Image_Reports']
            image_report_result = image_reports_collection.find_one({'image': self.image_record['image']})

            # Debug print statements to verify data structure
            print(f"Image Record: {self.image_record}")
            print(f"Image Report Result: {image_report_result}")

            # Display image and associated information
            self.display_image_and_text(image_path, self.image_record, image_report_result)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'An error occurred: {str(e)}')

    def display_image_and_text(self, image_path, image_record, image_report_result):
        # Load image from local path
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, 'Error', 'Could not load image.')
            return

        # Widget for displaying image
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)

        # Clear layouts before adding new widgets
        self.clear_layout(self.description_layout)
        self.clear_layout(self.case_topic_layout)

        # Add image to layout
        self.description_layout.addWidget(image_label)

        # Information from Image_Descriptions
        form_layout = QFormLayout()
        desc_label = QLabel('Description Information:')
        desc_label.setStyleSheet("font-weight: bold;")
        form_layout.addRow(desc_label)
        for key, value in image_record.items():
            if key == '_id':  # Ignore '_id' field
                continue
            if key == 'Description':
                form_layout.addRow(QLabel('Description:'))
                for desc_key, desc_value in value.items():
                    label = QLabel(str(desc_value))
                    label.setWordWrap(True)
                    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                    form_layout.addRow(QLabel(f"  {desc_key}:"), label)
            else:
                label = QLabel(str(value))
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                form_layout.addRow(QLabel(f"{key}:"), label)

        self.description_layout.addLayout(form_layout)

        # Update Case and Topic information
        self.update_case_topic_info(image_report_result)

    def update_case_topic_info(self, image_report_result):
        # Add 'Case' information to the Case and Topic information widget
        if image_report_result and 'Case' in image_report_result:
            case_info = image_report_result['Case']
            form_layout = QFormLayout()
            case_label = QLabel('Case Information:')
            case_label.setStyleSheet("font-weight: bold;")
            form_layout.addRow(case_label)
            
            for key, value in case_info.items():
                label = QLabel(str(value))
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                form_layout.addRow(QLabel(f"{key}:"), label)
            self.case_topic_layout.addLayout(form_layout)

        # Add a spacer item between Case and Topic information
        spacer_item = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.case_topic_layout.addItem(spacer_item)

        # Add 'Topic' information to the Case and Topic information widget
        if image_report_result and 'Topic' in image_report_result:
            topic_info = image_report_result['Topic']
            form_layout = QFormLayout()
            topic_label = QLabel('Topic Information:')
            topic_label.setStyleSheet("font-weight: bold;")
            form_layout.addRow(topic_label)
            for key, value in topic_info.items():
                label = QLabel(str(value))
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                form_layout.addRow(QLabel(f"{key}:"), label)
            self.case_topic_layout.addLayout(form_layout)

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


                
class ImageDownloadDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Select Image Type')
        layout = QVBoxLayout()

        self.ct_button = QPushButton('Download only CT images')
        self.ct_button.setStyleSheet("font-weight: bold;")
        self.ct_button.clicked.connect(lambda: self.download_images('TAC'))
        layout.addWidget(self.ct_button)

        self.mri_button = QPushButton('Download only MRI images')
        self.mri_button.setStyleSheet("font-weight: bold;")
        self.mri_button.clicked.connect(lambda: self.download_images('MRI'))
        layout.addWidget(self.mri_button)

        self.all_button = QPushButton('Download all images')
        self.all_button.setStyleSheet("font-weight: bold;")
        self.all_button.clicked.connect(lambda: self.download_images('ALL'))
        layout.addWidget(self.all_button)

        self.setLayout(layout)

    def sanitize_filename(self, filename):
        """Remove invalid characters from a filename."""
        return "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in filename)

    def download_images(self, image_type):
        # Path to the CSV file in the same directory as the executable
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file_path = os.path.join(script_dir, 'MedPix-2-0/images_overview.csv')  # Change this to the actual CSV file path

        # Read the CSV file
        image_urls = []
        try:
            with open(csv_file_path, newline='') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:
                    if image_type == 'ALL' or row['Type'] == image_type:
                        image_urls.append(row['Img_Large'])
        except FileNotFoundError:
            QMessageBox.warning(self, 'Error', f'CSV file not found at {csv_file_path}.')
            return
        except KeyError as e:
            QMessageBox.warning(self, 'Error', f'CSV file is missing expected column: {e}')
            return
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'An error occurred while reading the CSV file: {e}')
            return

        if not image_urls:
            QMessageBox.information(self, 'No Images', 'No images found for the selected type.')
            return

        # Create a folder dialog to choose the download location
        folder_dialog = QFileDialog()
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly)
        if folder_dialog.exec_() == QFileDialog.Accepted:
            download_folder = folder_dialog.selectedFiles()[0]

            # Download the images
            for url in image_urls:
                try:
                    response = requests.get(url)
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    image = Image.open(BytesIO(response.content))
                    file_name = self.sanitize_filename(url.split('/')[-1].split('.')[0]) + '.png'
                    file_path = os.path.join(download_folder, file_name)
                    image.save(file_path, 'PNG')
                except requests.exceptions.RequestException as e:
                    QMessageBox.warning(self, 'Error', f"Failed to download image from {url}. Error: {e}")
                except Exception as e:
                    QMessageBox.warning(self, 'Error', f"An error occurred while processing image from {url}. Error: {e}")

            QMessageBox.information(self, 'Download Complete', f"Images downloaded to {download_folder}")

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QScrollArea, QPushButton, QMessageBox

import json
from bson import ObjectId
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QScrollArea, QPushButton, QFileDialog, QMessageBox


class QueryResultsDialog(QDialog):
    def __init__(self, result_texts, query_results, db):
        super().__init__()

        self.db = db  # Store the database reference
        self.query_results = query_results

        self.setWindowTitle('Query Results')
        self.setMinimumSize(900, 900)
        layout = QVBoxLayout()

        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(result_texts)
        self.text_edit.setMinimumSize(300, 300)
        self.text_edit.setStyleSheet("font-weight: bold;")

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.text_edit)
        scroll_area.setWidgetResizable(True)
        
        layout.addWidget(scroll_area)

        save_button = QPushButton('Save to JSON')
        save_button.setStyleSheet("font-weight: bold;")
        save_button.clicked.connect(self.save_results_to_json)
        layout.addWidget(save_button)

        # Add dropdown menu and view button if query is on Image_Descriptions collection
        if all('image' in result for result in query_results):
            self.dropdown = QComboBox()
            self.dropdown.addItems(result['image'] for result in query_results)
            layout.addWidget(self.dropdown)

            view_button = QPushButton('View')
            view_button.setStyleSheet("font-weight: bold;")
            view_button.clicked.connect(self.view_image_info)
            layout.addWidget(view_button)

        self.setLayout(layout)

    def save_results_to_json(self):
        # Function to convert ObjectId to string
        def convert_objectid(obj):
            if isinstance(obj, ObjectId):
                return str(obj)
            raise TypeError

        # Convert results to handle ObjectId
        converted_results = []
        for result in self.query_results:
            if '_id' in result:
                result['_id'] = str(result['_id'])  # Convert ObjectId to string
            converted_results.append(result)
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save JSON file", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_path:
            with open(file_path, 'w') as json_file:
                json.dump(converted_results, json_file, indent=4, default=convert_objectid)
            QMessageBox.information(self, 'Save Confirmation', f'Results saved to {file_path}')

    def view_image_info(self):
        # Get the selected image ID
        image_id = self.dropdown.currentText()
        if not image_id:
            QMessageBox.warning(self, 'Warning', 'Please select an image ID.')
            return

        try:
            # Get the index of the selected image ID
            image_index = self.dropdown.currentIndex()
            # Open the image display window
            image_display_dialog = NewImageDisplayDialog(self.db, self.query_results, image_index)
            image_display_dialog.exec_()
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'An error occurred: {str(e)}')


class NewImageDisplayDialog(QDialog):
    def __init__(self, db, query_results, image_index=0):
        super().__init__()
        self.db = db
        self.query_results = query_results
        self.image_index = image_index
        self.setWindowTitle('Visualizza Immagine')
        self.setMinimumSize(800, 400)
        self.fullscreen = False  # fullscreen mode

        # general style
        self.setStyleSheet("""
            background-color: #E0FFFF;
            color: black;
            font-family: Arial;
            font-size: 16px;
        """)

        # fullscreen button in title bar
        self.fullscreen_button = QPushButton("Fullscreen", self)
        self.fullscreen_button.setStyleSheet("font-weight: bold;")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        title_bar_layout = QHBoxLayout()
        title_bar_layout.addWidget(self.fullscreen_button)
        title_bar_layout.addStretch()

        # navigation button
        self.previous_button = QPushButton("Back", self)
        self.previous_button.setStyleSheet("font-weight: bold;")
        self.previous_button.clicked.connect(self.show_previous_image)
        self.next_button = QPushButton("Next", self)
        self.next_button.setStyleSheet("font-weight: bold;")
        self.next_button.clicked.connect(self.show_next_image)
        navigation_layout = QHBoxLayout()
        navigation_layout.addWidget(self.previous_button)
        navigation_layout.addWidget(self.next_button)

        # saving button
        self.save_desc_button = QPushButton("Save Description JSON", self)
        self.save_desc_button.setStyleSheet("font-weight: bold;")
        self.save_desc_button.clicked.connect(self.save_description_to_json)
        self.save_case_topic_button = QPushButton("Save Case/Topic JSON", self)
        self.save_case_topic_button.setStyleSheet("font-weight: bold;")
        self.save_case_topic_button.clicked.connect(self.save_case_topic_to_json)
        save_buttons_layout = QHBoxLayout()
        save_buttons_layout.addWidget(self.save_desc_button)
        save_buttons_layout.addWidget(self.save_case_topic_button)

        # main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(title_bar_layout)
        main_layout.addLayout(navigation_layout)
        main_layout.addLayout(save_buttons_layout)

        content_layout = QHBoxLayout()

        # scroll area for images and description info
        description_scroll = QScrollArea()
        description_scroll.setWidgetResizable(True)
        self.description_widget = QWidget()
        self.description_layout = QVBoxLayout(self.description_widget)
        description_scroll.setWidget(self.description_widget)
        description_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        description_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content_layout.addWidget(description_scroll, 1)

        # scroll area for case and topic info
        case_topic_scroll = QScrollArea()
        case_topic_scroll.setWidgetResizable(True)
        self.case_topic_widget = QWidget()
        self.case_topic_layout = QVBoxLayout(self.case_topic_widget)
        case_topic_scroll.setWidget(self.case_topic_widget)
        case_topic_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        case_topic_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content_layout.addWidget(case_topic_scroll, 2)

        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

        # load and visualize data
        self.load_image_and_info()

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.showNormal()
        else:
            self.showFullScreen()
        self.fullscreen = not self.fullscreen

    def load_image_and_info(self):
        try:
            # get path of the current image
            image_record = self.query_results[self.image_index]
            image_path = rf"MedPix-2-0/{image_record['image']}"

            # get case and topic info from Image_Reports
            image_reports_collection = self.db['Image_Reports']
            image_report_result = image_reports_collection.find_one({'image': image_record['image']})

            # Debug print statements for data structure
            print(f"Image Record: {image_record}")
            print(f"Image Report Result: {image_report_result}")

            # clear layout
            self.clear_layout(self.description_layout)
            self.clear_layout(self.case_topic_layout)

            # visualization of image and related info
            self.display_image_and_text(image_path, image_record, image_report_result)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'An error occurred: {str(e)}')

    def display_image_and_text(self, image_path, image_record, image_report_result):
        # load image from local path
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, 'Error', f'An error occurred: {str(e)}')
            return

        # visualize image widget
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)

        # add image to layout
        self.description_layout.addWidget(image_label)

        # info from Image_Descriptions
        form_layout = QFormLayout()
        desc_label = QLabel('Description:')
        desc_label.setStyleSheet("font-weight: bold;")
        form_layout.addRow(desc_label)
        for key, value in image_record.items():
            if key == '_id':  # Ignora il campo '_id'
                continue
            if key == 'Description':
                form_layout.addRow(QLabel('Descrizione:'))
                for desc_key, desc_value in value.items():
                    label = QLabel(str(desc_value))
                    label.setWordWrap(True)
                    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                    form_layout.addRow(QLabel(f"  {desc_key}:"), label)
            else:
                label = QLabel(str(value))
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                form_layout.addRow(QLabel(f"{key}:"), label)

        self.description_layout.addLayout(form_layout)

        # update info about Case e Topic
        self.update_case_topic_info(image_report_result)

    def update_case_topic_info(self, image_report_result):
        # update info about Case to case and topic info widget
        if image_report_result and 'Case' in image_report_result:
            case_info = image_report_result['Case']
            form_layout = QFormLayout()
            case_label = QLabel('Case:')
            case_label.setStyleSheet("font-weight: bold;")
            form_layout.addRow(case_label)
            
            for key, value in case_info.items():
                label = QLabel(str(value))
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                form_layout.addRow(QLabel(f"{key}:"), label)
            self.case_topic_layout.addLayout(form_layout)

        # add space per visualization purposes between Case e Topic
        spacer_item = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.case_topic_layout.addItem(spacer_item)

        # update info about Topic to case and topic info widget
        if image_report_result and 'Topic' in image_report_result:
            topic_info = image_report_result['Topic']
            form_layout = QFormLayout()
            topic_label = QLabel('Topic:')
            topic_label.setStyleSheet("font-weight: bold;")
            form_layout.addRow(topic_label)
            for key, value in topic_info.items():
                label = QLabel(str(value))
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                form_layout.addRow(QLabel(f"{key}:"), label)
            self.case_topic_layout.addLayout(form_layout)

    def clear_layout(self, layout):
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                # manage nested layouts
                self.clear_layout(item.layout())

    def show_previous_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.load_image_and_info()

    def show_next_image(self):
        if self.image_index < len(self.query_results) - 1:
            self.image_index += 1
            self.load_image_and_info()

    def save_description_to_json(self):
        try:
            # get record of the current image
            image_record = self.query_results[self.image_index]
            # path for save JSON file
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Description JSON file", "", "JSON Files (*.json);;All Files (*)", options=options)
            if file_path:
                with open(file_path, 'w') as json_file:
                    json.dump(image_record, json_file, indent=4)
                QMessageBox.information(self, 'Save Confirmation', f'Description saved to {file_path}')
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'An arror occured during saving process: {str(e)}')

    def save_case_topic_to_json(self):
        try:
            # Estrai il record dell'immagine corrente
            image_record = self.query_results[self.image_index]
            # Recupera le informazioni su 'Case' e 'Topic' dalla vista 'Image_Reports'
            image_reports_collection = self.db['Image_Reports']
            image_report_result = image_reports_collection.find_one({'image': image_record['image']})
            # Chiedi all'utente il percorso dove salvare il file JSON
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Case/Topic JSON file", "", "JSON Files (*.json);;All Files (*)", options=options)
            if file_path:
                with open(file_path, 'w') as json_file:
                    json.dump(image_report_result, json_file, indent=4)
                QMessageBox.information(self, 'Save Confirmation', f'Case/Topic saved to {file_path}')
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'An arror occured during saving process: {str(e)}')



class FullScreenResultsWindow(QMainWindow):
    def __init__(self, results):
        super().__init__()

        self.setWindowTitle('Query Results')
        self.results_label = QLabel('\n'.join(str(result) for result in results))
        self.results_label.setStyleSheet("font-weight: bold;")
        self.setCentralWidget(self.results_label)

        self.showFullScreen()

class MongoDBQueryApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.query_inputs = {}
        self.description_expanded = False  # Track the state of the description inputs
        self.report_expanded = False  # Track the state of the report inputs
        self.case_expanded = False  # Track the state of the case inputs
        self.topic_expanded = False  # Track the state of the topic inputs

        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['Medical_Database']

    def initUI(self):
        self.setWindowTitle('Medical Database')

        main_layout = QVBoxLayout()

        title = QLabel('Medical Database')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #4A4A4A;")
        main_layout.addWidget(title)

        collection_layout = QHBoxLayout()
        self.image_radio = QRadioButton('Image_Descriptions')
        self.image_radio.setStyleSheet("font-weight: bold;")
        self.clinical_radio = QRadioButton('Clinical_Reports')
        self.clinical_radio.setStyleSheet("font-weight: bold;")
        self.complete_radio = QRadioButton('Image_Reports')
        self.complete_radio.setStyleSheet("font-weight: bold;")
        collection_layout.addWidget(self.image_radio)
        collection_layout.addWidget(self.clinical_radio)
        collection_layout.addWidget(self.complete_radio)
        main_layout.addLayout(collection_layout)

        self.image_radio.toggled.connect(self.update_key_fields)
        self.clinical_radio.toggled.connect(self.update_key_fields)
        self.complete_radio.toggled.connect(self.update_key_fields)

        self.key_fields_group = QGroupBox("Key Fields")
        self.key_fields_layout = QVBoxLayout()
        self.key_fields_group.setLayout(self.key_fields_layout)
        self.key_fields_group.setStyleSheet("background-color: #E0F7FA; border: 1px solid #B2EBF2; padding: 10px; font-weight: bold;")
        main_layout.addWidget(self.key_fields_group)

        self.query_input_layout = QFormLayout()
        self.query_input_group = QGroupBox("Query Inputs")
        self.query_input_group.setLayout(self.query_input_layout)
        self.query_input_group.setStyleSheet("background-color: #E0F7FA; border: 1px solid #B2EBF2; padding: 10px; font-weight: bold;")
        main_layout.addWidget(self.query_input_group)

        button_layout = QHBoxLayout()
        self.query_button = QPushButton('Query')
        self.query_button.setStyleSheet("font-weight: bold;")
        self.download_button = QPushButton('Download')
        self.download_button.setStyleSheet("font-weight: bold;")
        button_layout.addWidget(self.query_button)
        button_layout.addWidget(self.download_button)
        main_layout.addLayout(button_layout)
        
        self.view_image_button = QPushButton('View Image')
        self.view_image_button.setStyleSheet("font-weight: bold;")
        button_layout.addWidget(self.view_image_button)
        self.view_image_button.clicked.connect(self.show_view_image_dialog)

        self.query_button.clicked.connect(self.run_query)
        self.download_button.clicked.connect(self.open_download_dialog)

        self.setLayout(main_layout)

        self.resize(800, 800)

        self.setStyleSheet("background-color: #E0F7FA; font-weight: bold;")

    def update_key_fields(self):
        # remove existing widgets key fields layout
        for i in reversed(range(self.key_fields_layout.count())):
            widget = self.key_fields_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # remove existing widgets in query input layout
        for i in reversed(range(self.query_input_layout.count())):
            widget = self.query_input_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # get name of the selected collection
        collection_name = self.get_selected_collection()
        if not collection_name:
            return

        # get document from collection
        collection = self.db[collection_name]
        document = collection.find_one()

        if document:
            self.key_checkboxes = []
            for key in document.keys():
                # skip _id
                if key == '_id':
                    continue
                if collection_name == 'Image_Descriptions' and key == 'Description':
                    continue
                if collection_name == 'Clinical_Reports' and key == 'Report':
                    continue
                
                # create a checkbox for each key different from _id
                checkbox = QCheckBox(key)
                checkbox.setStyleSheet("font-weight: bold;")
                self.key_checkboxes.append(checkbox)
                checkbox.stateChanged.connect(self.update_query_inputs)
                self.key_fields_layout.addWidget(checkbox)

            if 'Description' in document:
                description_button = QCheckBox('Description')
                description_button.setStyleSheet("font-weight: bold;")
                description_button.clicked.connect(self.toggle_description_inputs)
                self.key_fields_layout.addWidget(description_button)

    def toggle_description_inputs(self):
        if self.description_expanded:
            self.hide_description_inputs()
        else:
            self.show_description_inputs()
        self.description_expanded = not self.description_expanded

    def show_description_inputs(self):
        collection_name = self.get_selected_collection()
        if collection_name:
            document = self.db[collection_name].find_one()
            if document and 'Description' in document and isinstance(document['Description'], dict):
                for key in document['Description'].keys():
                    if f'Description.{key}' not in self.query_inputs:
                        line_edit = QLineEdit(self)
                        line_edit.setStyleSheet("font-weight: bold;")
                        self.query_inputs[f'Description.{key}'] = line_edit
                        self.query_input_layout.addRow(QLabel(f'Insert Description.{key}:', self), line_edit)
                        if key == 'Sex':
                            line_edit.setPlaceholderText('(male or female)')

    def hide_description_inputs(self):
        for key in list(self.query_inputs.keys()):
            if key.startswith('Description.'):
                widget = self.query_inputs.pop(key)
                self.query_input_layout.removeRow(widget)

    def toggle_report_inputs(self):
        if self.report_expanded:
            self.hide_report_inputs()
        else:
            self.show_report_inputs()
        self.report_expanded = not self.report_expanded

    def show_report_inputs(self):
        collection_name = self.get_selected_collection()
        if collection_name:
            document = self.db[collection_name].find_one()
            if document and 'Report' in document and isinstance(document['Report'], dict):
                for key in document['Report'].keys():
                    if f'Report.{key}' not in self.query_inputs:
                        line_edit = QLineEdit(self)
                        line_edit.setStyleSheet("font-weight: bold;")
                        self.query_inputs[f'Report.{key}'] = line_edit
                        self.query_input_layout.addRow(QLabel(f'Insert Report.{key}:', self), line_edit)

    def hide_report_inputs(self):
        for key in list(self.query_inputs.keys()):
            if key.startswith('Report.'):
                widget = self.query_inputs.pop(key)
                self.query_input_layout.removeRow(widget)

    def toggle_case_inputs(self):
        if self.case_expanded:
            self.hide_case_inputs()
        else:
            self.show_case_inputs()
        self.case_expanded = not self.case_expanded

    def show_case_inputs(self):
        collection_name = self.get_selected_collection()
        if collection_name:
            document = self.db[collection_name].find_one()
            if document and 'Case' in document and isinstance(document['Case'], dict):
                for key in document['Case'].keys():
                    if f'Case.{key}' not in self.query_inputs:
                        line_edit = QLineEdit(self)
                        line_edit.setStyleSheet("font-weight: bold;")
                        self.query_inputs[f'Case.{key}'] = line_edit
                        self.query_input_layout.addRow(QLabel(f'Insert Case.{key}:', self), line_edit)

    def hide_case_inputs(self):
        for key in list(self.query_inputs.keys()):
            if key.startswith('Case.'):
                widget = self.query_inputs.pop(key)
                self.query_input_layout.removeRow(widget)

    def toggle_topic_inputs(self):
        if self.topic_expanded:
            self.hide_topic_inputs()
        else:
            self.show_topic_inputs()
        self.topic_expanded = not self.topic_expanded

    def show_topic_inputs(self):
        collection_name = self.get_selected_collection()
        if collection_name:
            document = self.db[collection_name].find_one()
            if document and 'Topic' in document and isinstance(document['Topic'], dict):
                for key in document['Topic'].keys():
                    if f'Topic.{key}' not in self.query_inputs:
                        line_edit = QLineEdit(self)
                        line_edit.setStyleSheet("font-weight: bold;")
                        self.query_inputs[f'Topic.{key}'] = line_edit
                        self.query_input_layout.addRow(QLabel(f'Insert Topic.{key}:', self), line_edit)

    def hide_topic_inputs(self):
        for key in list(self.query_inputs.keys()):
            if key.startswith('Topic.'):
                widget = self.query_inputs.pop(key)
                self.query_input_layout.removeRow(widget)

    def update_query_inputs(self):
        for i in reversed(range(self.query_input_layout.count())):
            widget = self.query_input_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        self.query_inputs = {}
        for checkbox in self.key_checkboxes:
            if checkbox.isChecked():
                key = checkbox.text()
                line_edit = QLineEdit(self)
                line_edit.setStyleSheet("font-weight: bold;")
                self.query_inputs[key] = line_edit
                self.query_input_layout.addRow(QLabel(f'Insert {key}:', self), line_edit)
                if key == 'Type':
                    line_edit.setPlaceholderText('(CT or MR)')  # add legend field and example
                if key == 'U_id':
                    line_edit.setPlaceholderText('ex: MPX1009')  # add legend field and example
                if key == 'image':
                    line_edit.setPlaceholderText('ex: MPX1009_synpic46283')  # add legend field and example

        if any(checkbox.text() == 'Description' and checkbox.isChecked() for checkbox in self.key_checkboxes):
            self.show_description_inputs()
        elif any(checkbox.text() == 'Report' and checkbox.isChecked() for checkbox in self.key_checkboxes):
            self.show_report_inputs()
        elif any(checkbox.text() == 'Case' and checkbox.isChecked() for checkbox in self.key_checkboxes):
            self.show_case_inputs()
        elif any(checkbox.text() == 'Topic' and checkbox.isChecked() for checkbox in self.key_checkboxes):
            self.show_topic_inputs()

    def run_query(self):
        collection_name = self.get_selected_collection()
        if not collection_name:
            QMessageBox.warning(self, 'Warning', 'Please select a collection.')
            return

        query = {}
        for key, input_field in self.query_inputs.items():
            query_value = input_field.text()
            if query_value:
                query[key] = {"$regex": query_value, "$options": "i"}

        if not query:
            QMessageBox.warning(self, 'Warning', 'Please enter at least one query value.')
            return

        collection = self.db[collection_name]
        results = list(collection.find(query))

        if results:
            # Exclude '_id' from each result
            for result in results:
                if '_id' in result:
                    del result['_id']
            result_texts = '\n'.join([str(result) for result in results])
            dialog = QueryResultsDialog(result_texts, results, self.db)  # Pass self.db as argument
            dialog.exec_()
        else:
            self.query_results = None
            QMessageBox.information(self, 'Query Results', 'No results found.')


    def show_query_results(self):
        if self.query_results:
            self.full_screen_results_window = FullScreenResultsWindow(self.query_results)

    def save_to_json(self, results):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save JSON file", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_path:
            with open(file_path, 'w') as json_file:
                json.dump(results, json_file, indent=4)
            QMessageBox.information(self, 'Save Confirmation', f'Results saved to {file_path}')

    def download_image(self):
        if not hasattr(self, 'query_results') or not self.query_results:
            QMessageBox.warning(self, 'Warning', 'No query results to download.')
            return

        QMessageBox.information(self, 'Download', 'Image downloaded successfully (stub).')

    def get_selected_collection(self):
        if self.image_radio.isChecked():
            return 'Image_Descriptions'
        elif self.clinical_radio.isChecked():
            return 'Clinical_Reports'
        elif self.complete_radio.isChecked():
            return 'Image_Reports'
        return None
    
    def open_download_dialog(self):
        download_dialog = ImageDownloadDialog()
        download_dialog.exec_()
        
    def show_view_image_dialog(self):
        view_image_dialog = ViewImageDialog(self.db)
        view_image_dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MongoDBQueryApp()
    ex.show()
    sys.exit(app.exec_())
