import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from utils.data_classes import Shelf, Product


@st.cache_resource
def load_models():
    shelf_model= YOLO("models\shelves_model.pt")
    product_model= YOLO("models\product_model.pt")
    return shelf_model, product_model


shelf_model, product_model=load_models()


# Detection functions
def detect_shelves(img):
    result=shelf_model(image,conf=0.58)
    shelves=result[0].obb.xyxyxyxy
    shelves_objs = [Shelf(shelf) for shelf in shelves]
    return shelves_objs

def detect_products(img):

    results = product_model(img)
    products = [Product(box) for box in results[0].boxes.xyxy]
    return products

# Drawing functions
def draw_shelf(img,shelf):
    shelf_points=np.array([shelf.p1,shelf.p2,shelf.p3,shelf.p4])
    cv2.polylines(img,[shelf_points],True,color=(0,0,222),thickness=3)
    return img

def draw_product(img, product):
    cv2.rectangle(img, product.p1, product.p2, color=(0,244,244),thickness=2)
    return img


#utility funcs

def calculate_shelf_area(shelf):

    points=np.array([shelf.p1,shelf.p2,shelf.p3,shelf.p4])
    return cv2.contourArea(points)


def calculate_product_area(product):

    width = abs(product.p2[0] - product.p1[0])
    height = abs(product.p2[1] - product.p1[1])
    return width*height



def estimate_additional_products(shelf, products_on_shelf):
    shelf_area = calculate_shelf_area(shelf)
    
    if not products_on_shelf:

        avg_product_area = shelf_area / 20  
    else:

        total_product_area = sum(calculate_product_area(p) for p in products_on_shelf)
        avg_product_area = total_product_area / len(products_on_shelf)
    
    remaining_area = shelf_area - sum(calculate_product_area(p) for p in products_on_shelf)
    additional_estimate = int(remaining_area / (avg_product_area * 1.2))  

    
    return max(0, additional_estimate)

def filter_products_on_shelf(products, shelf):

    shelf_poly = np.array([shelf.p1, shelf.p2, shelf.p3, shelf.p4], dtype=np.float32)
    
    products_on_shelf = []
    
    for product in products:

        center_x = (product.p1[0] + product.p2[0]) / 2.0
        center_y = (product.p1[1] + product.p2[1]) / 2.0
        

        if cv2.pointPolygonTest(shelf_poly, (float(center_x), float(center_y)), False) >= 0:
            products_on_shelf.append(product)
    
    return products_on_shelf

# Streamlit UI
st.title("Retail Shelf Monitoring System")

uploaded_file = st.file_uploader("Upload a store shelf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    shelves = detect_shelves(image)
    products = detect_products(image)

    

    st.subheader("Combined Detection")
    combined_img = image.copy()
    for shelf in shelves:
        combined_img = draw_shelf(combined_img, shelf)
    for product in products:
        combined_img = draw_product(combined_img, product)
    st.image(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    

    st.subheader("Shelf Insights")
    
    for i, shelf in enumerate(shelves):

        products_on_shelf = filter_products_on_shelf(products, shelf)
        


        product_count = len(products_on_shelf)
        additional_estimate = estimate_additional_products(shelf, products_on_shelf)
        


        with st.expander(f"Shelf {i+1} Analysis"):
            col1, col2 = st.columns(2)
            col1.metric("Products Detected", product_count)
            col2.metric("Estimated Additional Capacity", additional_estimate)
            

            shelf_img = image.copy()
            shelf_img = draw_shelf(shelf_img, shelf)
            for product in products_on_shelf:
                shelf_img = draw_product(shelf_img, product)
            st.image(cv2.cvtColor(shelf_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            

           