#!/usr/bin/python -W all
# link-articles.py: link meta data to newspaper article texts
# usage: link-articles.py (via cgi)
# 20180706 erikt(at)xs4all.nl

import cgi
import cgitb
import csv
import re
import sys
import xml.etree.ElementTree as ET
 
NEWSPAPERMETA = "05NRC Handelsblad"
NEWSPAPERXML = "00Algemeen-Handelsblad"
PAPERFIELD = "Titel krant"
DATEFIELD = "Datum"
PAGEFIELD = "Paginanummer"
XMLDIR = "/var/www/data"
METADATAFILE = XMLDIR+"/frank-dutch.csv"
SEPARATOR = ","

def readTexts(newspaper,date,page):
    xmlFileName = XMLDIR+"/"+newspaper+"-"+date+"-"+page+".xml"
    dataRoot = ET.parse(xmlFileName).getroot()
    dataOut = []
    for text in dataRoot:
        textData = ""
        for paragraph in text: textData += paragraph.text
        utfData = textData.encode("utf-8")
        utfData = re.sub(r'"',"''",utfData)
        dataOut.append(utfData)
    dataOut = sorted(dataOut,key=lambda s: len(s),reverse=True)
    return(dataOut)
 
def readMetaData(newspaper,date,page):
    inFile = open(METADATAFILE,"r")
    csvReader = csv.DictReader(inFile,delimiter=SEPARATOR)
    dataOut = []
    for row in csvReader:
        if row[PAPERFIELD] == newspaper and \
           row[DATEFIELD] == date and \
           row[PAGEFIELD] == page: dataOut.append(row)
    dataOut = sorted(dataOut,key=lambda r: float(r["Oppervlakte"]),reverse=True)
    return(dataOut)

def printData(newspaper,date,page,texts,metadata):
    minIndex = min(len(texts),len(metadata))
    print("<h2>"+newspaper+" "+date+" page "+page+"</h2>")
    print("<table>")
    for i in range(0,minIndex):
        shortText = texts[i][0:80]      
        print("<tr><td>"+metadata[i]["Afbeelding"])
        print("</td><td>"+metadata[i]["Soort Auteur"])
        print("</td><td>"+metadata[i]["Aard nieuws"])
        print("</td><td>"+metadata[i]["Genre"])
        print("</td><td>"+metadata[i]["Onderwerp"])
        print("</td><td>"+metadata[i]["Oppervlakte"])
        print("</td><td>"+str(len(texts[i])))
        print("</td><td><font title=\""+texts[i]+"\">"+shortText+"</font>")
        print("</td></tr>")
    print("</table>")
    if len(texts) > minIndex: print("<p>"+str(len(texts)-minIndex)+" more text")
    if len(metadata) > minIndex: print("<p>"+str(len(metadata)-minIndex)+" more metadata")
    return()

def main(argv):
    print("Content-Type: text/html\n")
    cgitb.enable()
    for page in [1,2,11]:
        year = str(1965)
        month = str(12)
        day = str(4)
        page = str(page)
        newspaper = NEWSPAPERMETA
        date = month+"/"+day+"/"+year
        metaData = readMetaData(newspaper,date,page)
        newspaper = NEWSPAPERXML
        if len(month) < 2: month = "0"+month
        if len(day) < 2: day = "0"+day
        date = year+month+day
        texts = readTexts(newspaper,date,page)
        printData(newspaper,date,page,texts,metaData)
    sys.exit(0)

if __name__ == "__main__":
    sys.exit(main(sys.argv))

