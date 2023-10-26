# Automated flood extent mapping

## Background: 

During emergencies, flood extent maps are used to estimate damage to buildings and infrastructure. While numerous organizations conduct flood extent assessments, obtaining readily available and clear flood extent maps is often challenging. The process is time-consuming, and the methodologies used are often unclear, making it difficult to assess the validity of the results in short time. MapAction recently developed a Flood Mapping Tool, that allows to estimate flood extent using Sentinel-1 synthetic-aperture radar SAR data. The methodology is based on a recommended practice published by the United Nations Platform for Space-based Information for Disaster Management and Emergency Response (UN-SPIDER) and their source code is open access. However, some challenges persist: no skill assessment was conducted for the methodology, and the outcomes rely on a free parameter (threshold for image comparison) that is not readily assessable in advance, which makes the tool less user-friendly for non-experts. 

## Goal: 

Create a Python application that takes as input a place and a date, gathers the relevant images (Sentinel 2) from the Copernicus Data Space Ecosystem and outputs a flood extent in vector format (geojson). 

## Desired outputs: 

- User-friendly Python code in a Jupyter Notebook, with few parameters as input (e.g., dates, location, etc.) and vector dataset as output. 
- Streamlit app, possibly incorporated into MapAction's app (with both organisations acknowledged) 