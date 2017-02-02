#!/usr/bin/env python

#> \file 
#> \author David Ladd, Reused: Hashem Yousefi 
#> \brief This is an example to use linear fitting to fit the beginning of linear heart tube.
#>
#> \section LICENSE
#>
#> Version: MPL 1.1/GPL 2.0/LGPL 2.1
#>
#> The contents of this file are subject to the Mozilla Public License
#> Version 1.1 (the "License"); you may not use this file except in
#> compliance with the License. You may obtain a copy of the License at
#> http://www.mozilla.org/MPL/
#>
#> Software distributed under the License is distributed on an "AS IS"
#> basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#> License for the specific language governing rights and limitations
#> under the License.
#>
#> The Original Code is OpenCMISS
#>
#> The Initial Developer of the Original Code is University of Auckland,
#> Auckland, New Zealand and University of Oxford, Oxford, United
#> Kingdom. Portions created by the University of Auckland and University
#> of Oxford are Copyright (C) 2007 by the University of Auckland and
#> the University of Oxford. All Rights Reserved.
#>
#> Contributor(s): 
#>
#> Alternatively, the contents of this file may be used under the terms of
#> either the GNU General Public License Version 2 or later (the "GPL"), or
#> the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
#> in which case the provisions of the GPL or the LGPL are applicable instead
#> of those above. if you wish to allow use of your version of this file only
#> under the terms of either the GPL or the LGPL, and not to allow others to
#> use your version of this file under the terms of the MPL, indicate your
#> decision by deleting the provisions above and replace them with the notice
#> and other provisions required by the GPL or the LGPL. if you do not delete
#> the provisions above, a recipient may use your version of this file under
#> the terms of any one of the MPL, the GPL or the LGPL.
#>
### to be used for segmenting embryonic heart and fitting with an initial meshes.
#<

import sys, os
import exfile
import numpy
from numpy import linalg
import math
import random

# Intialise OpenCMISS/iron 
from opencmiss.iron import iron

# defining the output file to be written in the ExDataFile
def writeExdataFile(filename,dataPointLocations,dataErrorVector,dataErrorDistance,offset):
    "Writes data points to an exdata file"

    numberOfDimensions = dataPointLocations[1].shape[0]
    try:
        f = open(filename,"w")
        if numberOfDimensions == 1:
            header = '''Group name: DataPoints
 #Fields=3
 1) data_coordinates, coordinate, rectangular cartesian, #Components='''+str(numberOfDimensions)+'''
  x.  Value index=1, #Derivatives=0, #Versions=1
 2) data_error, field, rectangular cartesian, #Components='''+str(numberOfDimensions)+'''
  x.  Value index=2, #Derivatives=0, #Versions=1
 3) data_distance, field, real, #Components=1
  1.  Value index=3, #Derivatives=0, #Versions=1
'''
        elif numberOfDimensions == 2:
            header = '''Group name: DataPoints
 #Fields=3
 1) data_coordinates, coordinate, rectangular cartesian, #Components='''+str(numberOfDimensions)+'''
  x.  Value index=1, #Derivatives=0, #Versions=1
  y.  Value index=2, #Derivatives=0, #Versions=1
 2) data_error, field, rectangular cartesian, #Components='''+str(numberOfDimensions)+'''
  x.  Value index=3, #Derivatives=0, #Versions=1
  y.  Value index=4, #Derivatives=0, #Versions=1
 3) data_distance, field, real, #Components=1
  1.  Value index=5, #Derivatives=0, #Versions=1
'''
        elif numberOfDimensions == 3:
             header = '''Group name: DataPoints
 #Fields=3
 1) data_coordinates, coordinate, rectangular cartesian, #Components='''+str(numberOfDimensions)+'''
  x.  Value index=1, #Derivatives=0, #Versions=1
  y.  Value index=2, #Derivatives=0, #Versions=1
  x.  Value index=3, #Derivatives=0, #Versions=1
 2) data_error, field, rectangular cartesian, #Components='''+str(numberOfDimensions)+'''
  x.  Value index=4, #Derivatives=0, #Versions=1
  y.  Value index=5, #Derivatives=0, #Versions=1
  z.  Value index=6, #Derivatives=0, #Versions=1
 3) data_distance, field, real, #Components=1
  1.  Value index=7, #Derivatives=0, #Versions=1
'''
        f.write(header)

        numberOfDataPoints = len(dataPointLocations)
        for i in range(numberOfDataPoints):
            line = " Node: " + str(offset+i+1) + '\n'
            f.write(line)
            for j in range (numberOfDimensions):
                line = ' ' + str(dataPointLocations[i,j]) + '\t'
                f.write(line)
            line = '\n'
            f.write(line)
            for j in range (numberOfDimensions):
                line = ' ' + str(dataErrorVector[i,j]) + '\t'
                f.write(line)
            line = '\n'
            f.write(line)
            line = ' ' + str(dataErrorDistance[i])
            f.write(line)
            line = '\n'
            f.write(line)
        f.close()
            
    except IOError:
        print ('Could not open file: ' + filename)

#=================================================================
# Control Panel
#=================================================================
# set the number of elements and the number of nodes for the cylinder 
numberOfDimensions = 3
numberOfGaussXi = 3 
numberOfGaussPointsPerFace = 4
numberOfCircumfrentialElementsPerQuarter = 2
numberOfCircumfrentialElements = 4*numberOfCircumfrentialElementsPerQuarter
numberOfCircumfrentialNodes = numberOfCircumfrentialElements
numberOfLengthElements = 8
numberOfLengthNodes = numberOfLengthElements+1
numberOfWallElements = 1
numberOfWallNodes = numberOfWallElements+1
meshOrigin = [0.0,0.0,0.0]
startpoint = 0
startIteration = 0
Epi = True
fixInterior = True
hermite = True

if startIteration > 1:
    exfileMesh = True
    exnode = exfile.Exnode("DeformedGeometry" + str(startIteration-1) + ".part0.exnode")
    exelem = exfile.Exelem("UndeformedGeometry.part0.exelem")
else:
    exfileMesh = False

'''
#  Reading data points from JSON files of Epi-cardial and Endo-cardial
if (Epi): 
    with open('EpiDataPoints.json') as data_file:
	    values = json.load(data_file)
else: 
    with open('EndoDataPoints.json') as data_file:
	    values = json.load(data_file)
oldNumberOfDataPoints = len(values)
print "oldNumberOfDataPoints = ", oldNumberOfDataPoints
'''
oldNumberOfDataPoints = 935

# Set Sobolev smoothing parameters and zero tolerance has been fixed 
tau = 0.00001
kappa = 0.00001
zeroTolerance = 0.00001



#=================================================================
#      CS and Region and Basis
#=================================================================
(coordinateSystemUserNumber,
    regionUserNumber,
    basisUserNumber,
    generatedMeshUserNumber,
    meshUserNumber,
    decompositionUserNumber,
    geometricFieldUserNumber,
    cylindricalGeometricFieldUserNumber,
    centerGeometricFieldUserNumber,
    equationsSetFieldUserNumber,
    dependentFieldUserNumber,
    independentFieldUserNumber,
    dataPointFieldUserNumber,
    materialFieldUserNumber,
    analyticFieldUserNumber,
    dependentDataFieldUserNumber,
    dataPointsUserNumber,
    dataProjectionUserNumber,
    equationsSetUserNumber,
    problemUserNumber) = range(1,21)

# Get the computational nodes information
#print dir(iron),'\n\n'
numberOfComputationalNodes = iron.ComputationalNumberOfNodesGet()
computationalNodeNumber = iron.ComputationalNodeNumberGet()

# Create a RC CS
coordinateSystem = iron.CoordinateSystem()
coordinateSystem.CreateStart(coordinateSystemUserNumber)
coordinateSystem.dimension = 3
coordinateSystem.CreateFinish()

# Create a region
region = iron.Region()
region.CreateStart(regionUserNumber,iron.WorldRegion)
region.label = "FittingRegion"
region.coordinateSystem = coordinateSystem
region.CreateFinish()

# define a basis 
basis = iron.Basis()
basis.CreateStart(basisUserNumber)
basis.type = iron.BasisTypes.LAGRANGE_HERMITE_TP
basis.numberOfXi = numberOfDimensions
if hermite:
    basis.interpolationXi = [iron.BasisInterpolationSpecifications.CUBIC_HERMITE]*3
else:
    basis.interpolationXi = [iron.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*3
basis.quadratureNumberOfGaussXi = [numberOfGaussXi]*3
basis.CreateFinish()

#
#print "CS, Region and basis setted up"
#


#=================================================================
# Mesh
#=================================================================

# creating the number of elements and the mesh origins ... and/or
# Start the creation of a manually generated mesh in the region
numberOfNodes = numberOfCircumfrentialElements*(numberOfLengthElements+1)*(numberOfWallElements+1)
numberOfElements = numberOfCircumfrentialElements*numberOfLengthElements
numberOfGaussPoints = numberOfElements*numberOfGaussPointsPerFace


print "numberOfElements = ", numberOfElements
print "numberOfNodes = ", numberOfNodes

if (exfileMesh):
    # Read previous mesh
    mesh = iron.Mesh()
    mesh.CreateStart(meshUserNumber, region, numberOfDimensions)
    mesh.NumberOfComponentsSet(1)
    mesh.NumberOfElementsSet(exelem.num_elements)
    # Define nodes for the mesh
    nodes = iron.Nodes()
    nodes.CreateStart(region, exnode.num_nodes)
    nodes.CreateFinish()
    # Define elements for the mesh
    elements = iron.MeshElements()
    meshComponentNumber = 1
    elements.CreateStart(mesh, meshComponentNumber, basis)
    for elem in exelem.elements:
        elements.NodesSet(elem.number, elem.nodes)
    elements.CreateFinish()
    mesh.CreateFinish()
else:
    mesh = iron.Mesh()
    mesh.CreateStart(meshUserNumber,region,3)
    mesh.origin = meshOrigin
    mesh.NumberOfComponentsSet(1)
    mesh.NumberOfElementsSet(numberOfElements)
# Define nodes for the mesh
    nodes = iron.Nodes()
    nodes.CreateStart(region,numberOfNodes)
    nodes.CreateFinish()
    elements = iron.MeshElements()
    meshComponentNumber = 1
    elements.CreateStart(mesh, meshComponentNumber, basis)
    elementNumber = 0
    for wallElementIdx in range(1,numberOfWallElements+1):
       for lengthElementIdx in range(1,numberOfLengthElements+1):
            for circumfrentialElementIdx in range(1,numberOfCircumfrentialElements+1):
                elementNumber = elementNumber + 1
                localNode1 = circumfrentialElementIdx + (lengthElementIdx - 1)*numberOfCircumfrentialElements + \
                    (wallElementIdx-1)*numberOfCircumfrentialNodes*numberOfLengthNodes
                if circumfrentialElementIdx == numberOfCircumfrentialElements:
                    localNode2 = 1 + (lengthElementIdx-1)*numberOfCircumfrentialNodes + \
                        (wallElementIdx-1)*numberOfCircumfrentialNodes*numberOfLengthNodes
                else: 
                    localNode2 = localNode1 + 1
                localNode3 = localNode1 + numberOfCircumfrentialNodes
                localNode4 = localNode2 + numberOfCircumfrentialNodes
                localNode5 = localNode1 + numberOfCircumfrentialNodes*numberOfLengthNodes
                localNode6 = localNode2 + numberOfCircumfrentialNodes*numberOfLengthNodes
                localNode7 = localNode3 + numberOfCircumfrentialNodes*numberOfLengthNodes
                localNode8 = localNode4 + numberOfCircumfrentialNodes*numberOfLengthNodes
                localNodes = [localNode1,localNode2,localNode3,localNode4,localNode5,localNode6,localNode7,localNode8]
#		print "Element Number = ",elementNumber
#         	print "Node numbers of the element", localNode1, localNode2, localNode3, localNode4, localNode5, localNode6, localNode7, localNode8 
                elements.NodesSet(elementNumber,localNodes)  
    elements.CreateFinish()
    mesh.CreateFinish() 


# Create a decomposition for the mesh
decomposition = iron.Decomposition()
decomposition.CreateStart(decompositionUserNumber,mesh)
decomposition.type = iron.DecompositionTypes.CALCULATED
decomposition.numberOfDomains = numberOfComputationalNodes
decomposition.CalculateFacesSet(True)
decomposition.CreateFinish()

#
#print "mesh decomposition finished"
#



                 #===== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ====# 
                 #
                 #        Geometric, Cylindrical Geometric and Centroid Geometric Fields          #
                 #
                 #===== ==== ===== ====== ===== ===== ===== ===== ===== ===== ===== ===== ===== ==#

# the location of  nodes for the mesh  
manualNodePoints = numpy.zeros((numberOfLengthNodes,numberOfCircumfrentialNodes,numberOfDimensions,numberOfWallNodes))


manualNodePoints[0,0,:,0] = [1020,510,120]
manualNodePoints[0,1,:,0] = [920,510,120]
manualNodePoints[0,2,:,0] = [810,505,120]
manualNodePoints[0,3,:,0] = [690,515,120]
manualNodePoints[0,4,:,0] = [610,523,120]
manualNodePoints[0,5,:,0] = [700,450,120]
manualNodePoints[0,6,:,0] = [800,410,120]
manualNodePoints[0,7,:,0] = [930,435,120]

manualNodePoints[1,0,:,0] = [1280,250,440]
manualNodePoints[1,1,:,0] = [1250,315,440]
manualNodePoints[1,2,:,0] = [1170,315,440]
manualNodePoints[1,3,:,0] = [1080,300,440]
manualNodePoints[1,4,:,0] = [1020,285,440]
manualNodePoints[1,5,:,0] = [1090,240,440]
manualNodePoints[1,6,:,0] = [1180,210,440]
manualNodePoints[1,7,:,0] = [1275,210,440]

manualNodePoints[2,0,:,0] = [1360,265,770]
manualNodePoints[2,1,:,0] = [1320,320,755]
manualNodePoints[2,2,:,0] = [1210,365,740]
manualNodePoints[2,3,:,0] = [1112,380,725]
manualNodePoints[2,4,:,0] = [1072,325,710]
manualNodePoints[2,5,:,0] = [1098,254,725]
manualNodePoints[2,6,:,0] = [1200,270,740]
manualNodePoints[2,7,:,0] = [1280,235,755]

manualNodePoints[3,0,:,0] = [1260,335,1040]
manualNodePoints[3,1,:,0] = [1220,376,995]
manualNodePoints[3,2,:,0] = [1177,397,950]
manualNodePoints[3,3,:,0] = [1142,410,897]
manualNodePoints[3,4,:,0] = [1076,319,852]
manualNodePoints[3,5,:,0] = [1090,208,897]
manualNodePoints[3,6,:,0] = [1155,237,943]
manualNodePoints[3,7,:,0] = [1213,273,996]

manualNodePoints[4,0,:,0] = [1123,270,1275]
manualNodePoints[4,1,:,0] = [1040,378,1053]
manualNodePoints[4,2,:,0] = [1000,350,960]
manualNodePoints[4,3,:,0] = [964,324,875]
manualNodePoints[4,4,:,0] = [977,260,741]
manualNodePoints[4,5,:,0] = [954,175,875]
manualNodePoints[4,6,:,0] = [995,200,960]
manualNodePoints[4,7,:,0] = [1035,238,1053]

manualNodePoints[5,0,:,0] = [860,320,1053]
manualNodePoints[5,1,:,0] = [850,394,950]
manualNodePoints[5,2,:,0] = [838,370,849]
manualNodePoints[5,3,:,0] = [766,390,716]
manualNodePoints[5,4,:,0] = [680,240,585]
manualNodePoints[5,5,:,0] = [680,240,710]
manualNodePoints[5,6,:,0] = [770,307,830]
manualNodePoints[5,7,:,0] = [770,307,950]

manualNodePoints[6,0,:,0] = [700,400,937]
manualNodePoints[6,1,:,0] = [667,480,934]
manualNodePoints[6,2,:,0] = [820,610,900]
manualNodePoints[6,3,:,0] = [650,650,860]
manualNodePoints[6,4,:,0] = [570,620,860]
manualNodePoints[6,5,:,0] = [530,500,877]
manualNodePoints[6,6,:,0] = [560,400,877]
manualNodePoints[6,7,:,0] = [620,340,917]

manualNodePoints[7,0,:,0] = [800,600,1140]
manualNodePoints[7,1,:,0] = [970,625,1170]
manualNodePoints[7,2,:,0] = [830,680,1140]
manualNodePoints[7,3,:,0] = [650,680,1110]
manualNodePoints[7,4,:,0] = [450,770,1080]
manualNodePoints[7,5,:,0] = [330,715,1050]
manualNodePoints[7,6,:,0] = [500,600,1080]
manualNodePoints[7,7,:,0] = [650,550,1110]

manualNodePoints[8,0,:,0] = [1080,480,1320]
manualNodePoints[8,1,:,0] = [1270,600,1320]
manualNodePoints[8,2,:,0] = [1070,620,1320]
manualNodePoints[8,3,:,0] = [820,590,1320]
manualNodePoints[8,4,:,0] = [600,625,1320]
manualNodePoints[8,5,:,0] = [410,650,1320]
manualNodePoints[8,6,:,0] = [550,490,1320]
manualNodePoints[8,7,:,0] = [810,460,1320]

# node positions of the outer surface ... 
manualNodePoints[0,0,:,1] = [1187,507,120]
manualNodePoints[0,1,:,1] = [1050,663,120]
manualNodePoints[0,2,:,1] = [821,584,120]
manualNodePoints[0,3,:,1] = [580,634,120]
manualNodePoints[0,4,:,1] = [487,520,120]
manualNodePoints[0,5,:,1] = [626,331,120]
manualNodePoints[0,6,:,1] = [880,225,120]
manualNodePoints[0,7,:,1] = [1206,238,120]

manualNodePoints[1,0,:,1] = [1571,260,453]
manualNodePoints[1,1,:,1] = [1333,394,440]
manualNodePoints[1,2,:,1] = [1104,483,440]
manualNodePoints[1,3,:,1] = [924,420,440]
manualNodePoints[1,4,:,1] = [697,334,440]
manualNodePoints[1,5,:,1] = [903,131,440]
manualNodePoints[1,6,:,1] = [1159,78,440]
manualNodePoints[1,7,:,1] = [1385,94,440]

manualNodePoints[2,0,:,1] = [1600,251,771]
manualNodePoints[2,1,:,1] = [1433,408,744]
manualNodePoints[2,2,:,1] = [1210,465,740]
manualNodePoints[2,3,:,1] = [1045,489,625]
manualNodePoints[2,4,:,1] = [900,350,572]
manualNodePoints[2,5,:,1] = [1021,206,606]
manualNodePoints[2,6,:,1] = [1173,114,738]
manualNodePoints[2,7,:,1] = [1323,135,766]

manualNodePoints[3,0,:,1] = [1419,326,1160]
manualNodePoints[3,1,:,1] = [1285,440,1010]
manualNodePoints[3,2,:,1] = [1137,522,876]
manualNodePoints[3,3,:,1] = [964,499,621]
manualNodePoints[3,4,:,1] = [853,309,593]
manualNodePoints[3,5,:,1] = [976,100,579]
manualNodePoints[3,6,:,1] = [1140,80,921]
manualNodePoints[3,7,:,1] = [1314,157,1046]

manualNodePoints[4,0,:,1] = [963,218,1541]
manualNodePoints[4,1,:,1] = [1074,470,1081]
manualNodePoints[4,2,:,1] = [1011,531,886]
manualNodePoints[4,3,:,1] = [865,503,555]
manualNodePoints[4,4,:,1] = [629,199,471]
manualNodePoints[4,5,:,1] = [781,91,711]
manualNodePoints[4,6,:,1] = [992,96,974]
manualNodePoints[4,7,:,1] = [1002,111,1271]

manualNodePoints[5,0,:,1] = [815,367,1315]
manualNodePoints[5,1,:,1] = [927,556,1040]
manualNodePoints[5,2,:,1] = [1012,577,954]
manualNodePoints[5,3,:,1] = [721,569,576]
manualNodePoints[5,4,:,1] = [359,380,396]
manualNodePoints[5,5,:,1] = [426,198,717]
manualNodePoints[5,6,:,1] = [659,140,907]
manualNodePoints[5,7,:,1] = [722,183,1138]

manualNodePoints[6,0,:,1] = [734,452,1085]
manualNodePoints[6,1,:,1] = [730,527,1056]
manualNodePoints[6,2,:,1] = [1021,638,954]
manualNodePoints[6,3,:,1] = [763,650,867]
manualNodePoints[6,4,:,1] = [461,657,758]
manualNodePoints[6,5,:,1] = [259,492,739]
manualNodePoints[6,6,:,1] = [400,339,908]
manualNodePoints[6,7,:,1] = [598,310,1094]

manualNodePoints[7,0,:,1] = [800,500,1145]
manualNodePoints[7,1,:,1] = [1137,493,1166]
manualNodePoints[7,2,:,1] = [1244,665,1117]
manualNodePoints[7,3,:,1] = [857,705,1075]
manualNodePoints[7,4,:,1] = [423,832,1015]
manualNodePoints[7,5,:,1] = [230,715,1080]
manualNodePoints[7,6,:,1] = [383,448,1109]
manualNodePoints[7,7,:,1] = [650,450,1150]

manualNodePoints[8,0,:,1] = [1114,446,1320]
manualNodePoints[8,1,:,1] = [1517,545,1320]
manualNodePoints[8,2,:,1] = [1212,693,1327]
manualNodePoints[8,3,:,1] = [823,652,1320]
manualNodePoints[8,4,:,1] = [477,739,1320]
manualNodePoints[8,5,:,1] = [186,765,1320]
manualNodePoints[8,6,:,1] = [448,493,1320]
manualNodePoints[8,7,:,1] = [807,440,1320]


#=================================================================
#   Calculating the derivaitves  and centers 
#=================================================================

#calculating the derivatives 
difference = numpy.zeros((numberOfLengthNodes,numberOfCircumfrentialNodes,numberOfDimensions,numberOfWallNodes))
differenceAverage = numpy.zeros((numberOfLengthNodes,numberOfCircumfrentialNodes,numberOfDimensions,numberOfWallNodes))
circumDeriv = numpy.zeros((numberOfLengthNodes,numberOfCircumfrentialNodes,numberOfDimensions,numberOfWallNodes))
directDeriv = numpy.zeros((numberOfLengthNodes,numberOfCircumfrentialNodes,numberOfDimensions,numberOfWallNodes))
lengthDeriv = numpy.zeros((numberOfLengthNodes,numberOfCircumfrentialNodes,numberOfDimensions,numberOfWallNodes))
#circumferential derivative to be calculated 
for k in range (numberOfWallNodes):
    for j in range (numberOfLengthNodes):
        for i in range (numberOfCircumfrentialNodes):
            if (i<7):
                for m in range (numberOfDimensions):
                    difference[j,i,m,k]=manualNodePoints[j,i+1,m,k]-manualNodePoints[j,i,m,k]
            else:
                for m in range (numberOfDimensions):
                    difference[j,i,m,k]=manualNodePoints[j,0,m,k]-manualNodePoints[j,7,m,k]
for k in range (numberOfWallNodes):
    for j in range (numberOfLengthNodes):
        for i in range (numberOfCircumfrentialNodes):
            if (i<7):
                for m in range (numberOfDimensions):
                    differenceAverage[j,i+1,m,k]=(difference[j,i+1,m,k]+difference[j,i,m,k])/2
            else:
                for m in range (numberOfDimensions):
                    differenceAverage[j,0,m,k]=(difference[j,0,m,k]+difference[j,7,m,k])/2
for k in range (numberOfWallNodes):
    for j in range (numberOfLengthNodes):
        for i in range (numberOfCircumfrentialNodes):
            for m in range (numberOfDimensions):
                circumDeriv[j,i,m,k]=differenceAverage[j,i,m,k]/math.sqrt(math.pow(differenceAverage[j,i,0,k],2) + math.pow(differenceAverage[j,i,1,k],2) + math.pow(differenceAverage[j,i,2,k],2))
# derivative of the length direction
for k in range (numberOfWallNodes):
    for i in range (numberOfCircumfrentialNodes):
        for j in range (numberOfLengthNodes):
            if (j<numberOfLengthNodes-1):
                for m in range (numberOfDimensions):
                    difference[j,i,m,k]=manualNodePoints[j+1,i,m,k]-manualNodePoints[j,i,m,k]
            else:
                for m in range (numberOfDimensions):
                    difference[j,i,m,k]=manualNodePoints[j,i,m,k]-manualNodePoints[j-1,i,m,k]
for k in range (numberOfWallNodes):
    for i in range (numberOfCircumfrentialNodes):
        for j in range (numberOfLengthNodes):
            if (j == 0):
                for m in range (numberOfDimensions): 
                    differenceAverage[j,i,m,k]=difference[j,i,m,k]
            if (j<numberOfLengthNodes-1):
                for m in range (numberOfDimensions):
                    differenceAverage[j+1,i,m,k]=(difference[j,i,m,k]+difference[j+1,i,m,k])/2
            else:
                for m in range (numberOfDimensions):
                    differenceAverage[j,i,m,k]=difference[j-1,i,m,k]
for k in range (numberOfWallNodes):
    for j in range (numberOfLengthNodes):
        for i in range (numberOfCircumfrentialNodes):
            for m in range (numberOfDimensions):
                lengthDeriv[j,i,m,k]=differenceAverage[j,i,m,k]/math.sqrt(math.pow(differenceAverage[j,i,0,k],2) + math.pow(differenceAverage[j,i,1,k],2) + math.pow(differenceAverage[j,i,2,k],2))
# the derivatives of the wall direction is defined in the below lines ... 
for i in range (numberOfCircumfrentialNodes):
    for j in range (numberOfLengthNodes):
        for m in range (numberOfDimensions):
            for k in range (numberOfWallNodes):
                difference[j,i,m,k] = manualNodePoints[j,i,m,1] - manualNodePoints[j,i,m,0]
for i in range (numberOfCircumfrentialNodes):
    for j in range (numberOfLengthNodes):

        for k in range (numberOfWallNodes):
            for m in range (numberOfDimensions):
                directDeriv[j,i,m,k] = difference[j,i,m,k]/math.sqrt(math.pow(difference[j,i,0,k],2) + math.pow(difference[j,i,1,k],2) + math.pow(difference[j,i,2,k],2))


#  Calculating the centers  from the average of the nodes   
center = numpy.zeros((numberOfLengthNodes,numberOfCircumfrentialNodes,numberOfDimensions,numberOfWallNodes))
centerAvg = numpy.zeros((numberOfLengthNodes,numberOfDimensions,numberOfWallNodes))

for k in range (numberOfWallNodes):
    for j in range (numberOfLengthNodes):
        for m in range (numberOfDimensions):
            for i in range (numberOfCircumfrentialNodes):
                centerAvg[j,m,k] = centerAvg[j,m,k] + manualNodePoints[j,i,m,k]
            centerAvg[j,m,k] = (centerAvg[j,m,k]/numberOfCircumfrentialNodes)

for k in range (numberOfWallNodes):
    for j in range (numberOfLengthNodes):
        for m in range (numberOfDimensions):
            for i in range (numberOfCircumfrentialNodes):
                center[j,i,m,k] = centerAvg[j,m,k]


#  cylindrical coordinate systems ...  

# Radius  
cylindricalNodePoints = numpy.zeros((numberOfLengthNodes,numberOfCircumfrentialNodes,numberOfDimensions,numberOfWallNodes))
if (k == 0):
    for j in range (numberOfLengthNodes):
        for i in range (numberOfCircumfrentialNodes):
            radius = random.randint(100,150)
            cylindricalNodePoints[j,i,0,k] = radius 
else:
    for j in range (numberOfLengthNodes):
        for i in range (numberOfCircumfrentialNodes):
            radius = random.randint(200,300)
            cylindricalNodePoints[j,i,0,k] = radius

# Theta 
for k in range (numberOfWallNodes):
    for j in range (numberOfLengthNodes):
        for i in range (numberOfCircumfrentialNodes):
           cylindricalNodePoints[j,i,1,k] = 45*i+22.5

#Z_0
for k in range (numberOfWallNodes):
    for j in range (numberOfLengthNodes):
        for i in range (numberOfCircumfrentialNodes):
           cylindricalNodePoints[j,i,2,k] = 100*j


# Create a field for the geometry
geometricField = iron.Field()
geometricField.CreateStart(geometricFieldUserNumber,region)
geometricField.meshDecomposition = decomposition
for dimension in range(numberOfDimensions):
    geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,dimension+1,1)
geometricField.ScalingTypeSet(iron.FieldScalingTypes.ARITHMETIC_MEAN)
geometricField.CreateFinish()


# Create a field for the cylinder geometry
cylindricalGeometricField = iron.Field()
cylindricalGeometricField.CreateStart(cylindricalGeometricFieldUserNumber,region)
cylindricalGeometricField.VariableLabelSet(iron.FieldVariableTypes.U,"CylindericalCoordinates")
cylindricalGeometricField.meshDecomposition = decomposition
for dimension in range(3):
    cylindricalGeometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,dimension+1,1)
cylindricalGeometricField.ScalingTypeSet(iron.FieldScalingTypes.ARITHMETIC_MEAN)
cylindricalGeometricField.CreateFinish()


# Create a field for the centroid
centerGeometricField = iron.Field()
centerGeometricField.CreateStart(centerGeometricFieldUserNumber,region)
centerGeometricField.VariableLabelSet(iron.FieldVariableTypes.U,"CentroidCoordinates")
centerGeometricField.meshDecomposition = decomposition
for dimension in range(3):
    centerGeometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,dimension+1,1)
centerGeometricField.ScalingTypeSet(iron.FieldScalingTypes.ARITHMETIC_MEAN)
centerGeometricField.CreateFinish()


# Get nodes
nodes = iron.Nodes()
region.NodesGet(nodes)
numberOfNodes = nodes.numberOfNodes

# Get or calculate geometric parameters
if (exfileMesh):
    # Read the geometric field from the exnode file
    for node_num in range(1, exnode.num_nodes + 1):
        for derivative in range(1,9):
            version = 1
            for component in range(1, numberOfDimensions + 1):
                component_name = ["x", "y", "z"][component - 1]
                value = exnode.node_value("Coordinate", component_name, node_num, derivative)
                geometricField.ParameterSetUpdateNode(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                      version, derivative, node_num, component, value)
    geometricField.ParameterSetUpdateStart(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)
    geometricField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)
else:
    # Create the geometric field
    for wallNodeIdx in range(1,numberOfWallNodes+1):
        for lengthNodeIdx in range(1,numberOfLengthNodes+1):
            for circumfrentialNodeIdx in range(1,numberOfCircumfrentialNodes+1):
                nodeNumber = circumfrentialNodeIdx + (lengthNodeIdx-1)*numberOfCircumfrentialNodes + (wallNodeIdx-1)*numberOfCircumfrentialNodes*numberOfLengthNodes 
                x = manualNodePoints[lengthNodeIdx-1, circumfrentialNodeIdx-1, 0, wallNodeIdx-1]
                y = manualNodePoints[lengthNodeIdx-1, circumfrentialNodeIdx-1, 1, wallNodeIdx-1]
                z = manualNodePoints[lengthNodeIdx-1, circumfrentialNodeIdx-1, 2, wallNodeIdx-1]
                xtangent = circumDeriv[lengthNodeIdx-1, circumfrentialNodeIdx-1, 0, wallNodeIdx-1]
                ytangent = circumDeriv[lengthNodeIdx-1, circumfrentialNodeIdx-1, 1, wallNodeIdx-1]
                ztangent = circumDeriv[lengthNodeIdx-1, circumfrentialNodeIdx-1, 2, wallNodeIdx-1]
                xnormal = directDeriv[lengthNodeIdx-1, circumfrentialNodeIdx-1, 0, wallNodeIdx-1]
                ynormal = directDeriv[lengthNodeIdx-1, circumfrentialNodeIdx-1, 1, wallNodeIdx-1]
                znormal = directDeriv[lengthNodeIdx-1, circumfrentialNodeIdx-1, 2, wallNodeIdx-1]
                zxnormal = lengthDeriv[lengthNodeIdx-1, circumfrentialNodeIdx-1, 0, wallNodeIdx-1]
                zynormal = lengthDeriv[lengthNodeIdx-1, circumfrentialNodeIdx-1, 1, wallNodeIdx-1]
                zznormal = lengthDeriv[lengthNodeIdx-1, circumfrentialNodeIdx-1, 2, wallNodeIdx-1]
                geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,1,nodeNumber,1,x)
                geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,1,nodeNumber,2,y)
                geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,1,nodeNumber,3,z)
                geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,1,xtangent)
                geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,2,ytangent)
                geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,3,ztangent)
                geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,1,zxnormal)
                geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,2,zynormal)
                geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,3,zznormal)
                geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S3,nodeNumber,1,xnormal)
                geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S3,nodeNumber,2,ynormal)
                geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S3,nodeNumber,3,znormal)    


    for wallNodeIdx in range(1,numberOfWallNodes+1):
        for lengthNodeIdx in range(1,numberOfLengthNodes+1):
            for circumfrentialNodeIdx in range(1,numberOfCircumfrentialNodes+1):
                nodeNumber = circumfrentialNodeIdx + (lengthNodeIdx-1)*numberOfCircumfrentialNodes + (wallNodeIdx-1)*numberOfCircumfrentialNodes*numberOfLengthNodes 
                r = cylindricalNodePoints[lengthNodeIdx-1, circumfrentialNodeIdx-1, 0, wallNodeIdx-1]
                theta = cylindricalNodePoints[lengthNodeIdx-1, circumfrentialNodeIdx-1, 1, wallNodeIdx-1]
                z_0 = cylindricalNodePoints[lengthNodeIdx-1, circumfrentialNodeIdx-1, 2, wallNodeIdx-1]
                cylindricalGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,1,nodeNumber,1,r)
                cylindricalGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,1,nodeNumber,2,theta)
                cylindricalGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,1,nodeNumber,3,z_0)  


    for wallNodeIdx in range(1,numberOfWallNodes+1):
        for lengthNodeIdx in range(1,numberOfLengthNodes+1):
            for circumfrentialNodeIdx in range(1,numberOfCircumfrentialNodes+1):
                nodeNumber = circumfrentialNodeIdx + (lengthNodeIdx-1)*numberOfCircumfrentialNodes + (wallNodeIdx-1)*numberOfCircumfrentialNodes*numberOfLengthNodes 
                cx = center[lengthNodeIdx-1, circumfrentialNodeIdx-1, 0, wallNodeIdx-1]
                cy = center[lengthNodeIdx-1, circumfrentialNodeIdx-1, 1, wallNodeIdx-1]
                cz = center[lengthNodeIdx-1, circumfrentialNodeIdx-1, 2, wallNodeIdx-1]
                centerGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,1,nodeNumber,1,cx)
                centerGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,1,nodeNumber,2,cy)
                centerGeometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                        1,1,nodeNumber,3,cz)
                

    # Update the geometric field
    geometricField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
    geometricField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)

    cylindricalGeometricField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
    cylindricalGeometricField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)

    centerGeometricField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
    centerGeometricField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)


# sys.exit()



#=================================================================
# Data Points
#=================================================================
numberOfDataPoints = oldNumberOfDataPoints

newDataPoints = numpy.loadtxt('Epinewdatapoints.txt', delimiter=' ', skiprows=1)

numberOfDataPoints = 0*numberOfDataPoints+newDataPoints.shape[0]
print oldNumberOfDataPoints, newDataPoints.shape[0], numberOfDataPoints
# Create the data points
dataPoints = iron.DataPoints()
dataPoints.CreateStart(dataPointsUserNumber,region,numberOfDataPoints)
dataPointLocations = numpy.zeros((numberOfDataPoints,3))
'''
for i in range (numberOfCircumfrentialNodes):
    for j in range (numberOfLengthNodes):
        dataPointLocations[i+j*8,:] = manualNodePoints[j,i,:,1]

print("Number of data points: " + str(numberOfDataPoints))
# reading from a text file containing the point clouds   
if (Epi): 
    with open("EpiDataPoints.txt", "r") as ins:
	    arrayOfInputData = []
	    for line in ins:
		    arrayOfInputData.append(line)
else: 
    with open("EndoDataPoints.txt", "r") as ins:
	    arrayOfInputData = []
	    for line in ins:
		    arrayOfInputData.append(line)
x = 0.0
y = 0.0
z = 0.0
#for i in range (startpoint, numberOfDataPoints + startpoint):
for i in range (oldNumberOfDataPoints):
	for j in range (5):
		sample = arrayOfInputData[i*5 + j]
		if (math.fmod(j,5) == 1):
			x = float (sample[12:25])				
		elif (math.fmod(j,5) == 2):
			y = float (sample[12:25])
		elif (math.fmod(j,5) == 3):
			z = float (sample[12:17])
#		dataPointLocations[i - startpoint,:] = [x,y,z]
		dataPointLocations[i,:] = [x,y,z]
# Set up data points with geometric values
#for dataPoint in range(oldNumberOfDataPoints):
#    dataPointId = dataPoint + 1
#    dataList = dataPointLocations[dataPoint,:]
#    dataPoints.PositionSet(dataPointId,dataList)
'''
dataPointLocations = newDataPoints
dataPointId = 1
for row in newDataPoints:
    print dataPointId,list(row)
    dataPoints.PositionSet(dataPointId,list(row))
    dataPointId = dataPointId + 1
dataPoints.CreateFinish()
 
if True:
    # Export undeformed mesh geometry
    print("Writing undeformed geometry")
    fields = iron.Fields()
    fields.CreateRegion(region)
    fields.NodesExport("UndeformedGeometryMF","FORTRAN")
    fields.ElementsExport("UndeformedGeometryMF","FORTRAN")
    fields.Finalise()
#=================================================================
# Data Projection on Geometric Field
#=================================================================
print("Projecting data points onto geometric field")
candidateElements = range(1,numberOfElements+1)
candidateFaceNormals = iron.ElementNormalXiDirections.PLUS_XI3*numpy.ones(numberOfElements,dtype=numpy.int32)
# Set up data projection
dataProjection = iron.DataProjection()
dataProjection.CreateStart(dataProjectionUserNumber,dataPoints,geometricField,iron.FieldVariableTypes.U)
#dataProjection.projectionType = iron.DataProjectionProjectionTypes.ALL_ELEMENTS
dataProjection.projectionType = iron.DataProjectionProjectionTypes.BOUNDARY_FACES
dataProjection.ProjectionCandidateFacesSet(candidateElements,candidateFaceNormals)
#dataProjection.ProjectionDataCandidateFacesSet([1,2,3],[1,2],[iron.ElementNormalXiDirections.PLUS_XI3,iron.ElementNormalXiDirections.PLUS_XI3])
dataProjection.CreateFinish()

# Evaluate data projection based on geometric field
dataProjection.DataPointsProjectionEvaluate(iron.FieldParameterSetTypes.VALUES)
# Create mesh topology for data projection
mesh.TopologyDataPointsCalculateProjection(dataProjection)
# Create decomposition topology for data projection
decomposition.TopologyDataProjectionCalculate()

# Cancel some projections
dataProjection.ProjectionCancelByDistance(iron.DataProjectionDistanceRelations.GREATER_EQUAL,50.0)

# Output data projection results
dataProjection.ResultAnalysisOutput("ProjectionAnalysis")

rmsError=dataProjection.ResultRMSErrorGet()
print("RMS error = "+ str(rmsError))

# Output the .exdata file.                                           
dataErrorVector = numpy.zeros((numberOfDataPoints,3))
dataErrorDistance = numpy.zeros(numberOfDataPoints)
for elementIdx in range(1,numberOfElements+1):
    numberOfProjectedDataPoints = decomposition.TopologyNumberOfElementDataPointsGet(elementIdx)
    for dataPointIdx in range(1,numberOfProjectedDataPoints+1):
        dataPointNumber = decomposition.TopologyElementDataPointUserNumberGet(elementIdx,dataPointIdx)
        errorVector = dataProjection.ResultProjectionVectorGet(dataPointNumber,3)
        dataErrorVector[dataPointNumber-1,0]=errorVector[0]
        dataErrorVector[dataPointNumber-1,1]=errorVector[1]
        dataErrorVector[dataPointNumber-1,2]=errorVector[2]
        errorDistance = dataProjection.ResultDistanceGet(dataPointNumber)
        dataErrorDistance[dataPointNumber-1]=errorDistance
 
# write data points to exdata file for CMGUI
offset = 0
writeExdataFile("DataPoints.part"+str(computationalNodeNumber)+".exdata",dataPointLocations,dataErrorVector,dataErrorDistance,offset)

print("Projection complete")
#exit(0)
 
#=================================================================
# Equations Set
#=================================================================
# Create vector fitting equations set
equationsSetField = iron.Field()
equationsSet = iron.EquationsSet()
equationsSetSpecification = [iron.EquationsSetClasses.FITTING,
                             iron.EquationsSetTypes.DATA_FITTING_EQUATION,
                             iron.EquationsSetSubtypes.DATA_POINT_FITTING, 
 			     iron.EquationsSetFittingSmoothingTypes.SOBOLEV_VALUE]
equationsSet.CreateStart(equationsSetUserNumber,region,geometricField,
        equationsSetSpecification, equationsSetFieldUserNumber, equationsSetField)
equationsSet.CreateFinish()

#=================================================================
# Dependent Field
#=================================================================
# Create dependent field (will be deformed fitted values based on data point locations)
dependentField = iron.Field()
equationsSet.DependentCreateStart(dependentFieldUserNumber,dependentField)
dependentField.VariableLabelSet(iron.FieldVariableTypes.U,"Dependent")
dependentField.ScalingTypeSet(iron.FieldScalingTypes.ARITHMETIC_MEAN)
dependentField.NumberOfComponentsSet(iron.FieldVariableTypes.U,numberOfDimensions)
dependentField.NumberOfComponentsSet(iron.FieldVariableTypes.DELUDELN,numberOfDimensions)
equationsSet.DependentCreateFinish()
# Initialise dependent field
dependentField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,0.0)

# Initialise dependent field to undeformed geometric field
for component in range (1,numberOfDimensions+1):
    geometricField.ParametersToFieldParametersComponentCopy(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                            component, dependentField, iron.FieldVariableTypes.U,
                                                            iron.FieldParameterSetTypes.VALUES, component)

#=================================================================
# Independent Field
#=================================================================
# Create data point field (independent field, with vector values stored at the data points)
independentField = iron.Field()
equationsSet.IndependentCreateStart(independentFieldUserNumber,independentField)
independentField.VariableLabelSet(iron.FieldVariableTypes.U,"data point vector")
independentField.VariableLabelSet(iron.FieldVariableTypes.V,"data point weight")
independentField.DataProjectionSet(dataProjection)
equationsSet.IndependentCreateFinish()
# Initialise data point vector field to 0
#independentField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,0.0)
# Initialise data point weight field to 1
#independentField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.V,iron.FieldParameterSetTypes.VALUES,1,1.0)
# loop over each element's data points and set independent field values to data point locations on surface of the sphere
for element in range(numberOfElements):
    elementId = element + 1
    elementDomain = decomposition.ElementDomainGet(elementId)
    if (elementDomain == computationalNodeNumber):
        numberOfProjectedDataPoints = decomposition.TopologyNumberOfElementDataPointsGet(elementId)
        for dataPoint in range(numberOfProjectedDataPoints):
            dataPointId = dataPoint + 1
            dataPointNumber = decomposition.TopologyElementDataPointUserNumberGet(elementId,dataPointId)
            dataList = dataPoints.PositionGet(dataPointNumber,3)
            # set data point field values
            for component in range(numberOfDimensions):
                componentId = component + 1
                dataPointNumberIndex = dataPointNumber - 1
                value = dataList[component]
                independentField.ParameterSetUpdateElementDataPointDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,elementId,dataPointId,componentId,value)

#=================================================================
# Material Field
#=================================================================
# Create material field (Sobolev parameters)
materialField = iron.Field()
equationsSet.MaterialsCreateStart(materialFieldUserNumber,materialField)
materialField.VariableLabelSet(iron.FieldVariableTypes.U,"Smoothing Parameters")
equationsSet.MaterialsCreateFinish()
# Set kappa and tau - Sobolev smoothing parameters
materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,tau)
materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,2,kappa)

#=================================================================
# Equations
#=================================================================
# Create equations
equations = iron.Equations()
equationsSet.EquationsCreateStart(equations)
equations.sparsityType = iron.EquationsSparsityTypes.FULL
equations.outputType = iron.EquationsOutputTypes.NONE
equationsSet.EquationsCreateFinish()

#=================================================================
# Problem setup
#=================================================================
# Create fitting problem
problem = iron.Problem()
problemSpecification = [iron.ProblemClasses.FITTING,
                        iron.ProblemTypes.DATA_FITTING,
                        iron.ProblemSubtypes.STATIC_FITTING]
problem.CreateStart(problemUserNumber, problemSpecification)
problem.CreateFinish()

# Create control loops
problem.ControlLoopCreateStart()
problem.ControlLoopCreateFinish()

# Create problem solver
solver = iron.Solver()
problem.SolversCreateStart()
problem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,solver)
solver.outputType = iron.SolverOutputTypes.NONE # NONE / MATRIX
#solver.outputType = iron.SolverOutputTypes.MATRIX # NONE / MATRIX
solver.linearType = iron.LinearSolverTypes.ITERATIVE
#solver.linearType = iron.LinearSolverTypes.DIRECT
#solver.LibraryTypeSet(iron.SolverLibraries.UMFPACK) # UMFPACK/SUPERLU
#solver.LibraryTypeSet(iron.SolverLibraries.MUMPS)
solver.linearIterativeAbsoluteTolerance = 1.0E-10
solver.linearIterativeRelativeTolerance = 1.0E-05
problem.SolversCreateFinish()

# Create solver equations and add equations set to solver equations
solver = iron.Solver()
solverEquations = iron.SolverEquations()
problem.SolverEquationsCreateStart()
problem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,solver)
solver.SolverEquationsGet(solverEquations)
#solverEquations.sparsityType = iron.SolverEquationsSparsityTypes.FULL
solverEquations.sparsityType = iron.SolverEquationsSparsityTypes.SPARSE
equationsSetIndex = solverEquations.EquationsSetAdd(equationsSet)
problem.SolverEquationsCreateFinish()

#=================================================================
# Boundary Conditions
#=================================================================


# Create boundary conditions and set first and last nodes to 0.0 and 1.0
boundaryConditions = iron.BoundaryConditions()
solverEquations.BoundaryConditionsCreateStart(boundaryConditions)

# for nodeIdx in range(numberOfLengthNodes*numberOfCircumfrentialNodes+1,numberOfLengthNodes*numberOfCircumfrentialNodes*2+1):
for nodeIdx in range(1,numberOfLengthNodes*numberOfCircumfrentialNodes+1):
    for componentIdx in range(1,4):
        for derivativeIdx in range(1,9):
            boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,
                                       1,derivativeIdx,nodeIdx,componentIdx,
                                       iron.BoundaryConditionsTypes.FIXED,0.0)
#for nodeIdx in range(5,9):
#    for componentIdx in range(1,4):
#        boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,
#                                   1,iron.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeIdx,componentIdx,
#                                   iron.BoundaryConditionsTypes.FIXED,0.0)
#        boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,
#                                   1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S3,nodeIdx,componentIdx,
#                                   iron.BoundaryConditionsTypes.FIXED,0.0)
#        boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,
#                                   1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2,nodeIdx,componentIdx,
#                                   iron.BoundaryConditionsTypes.FIXED,0.0)
#        boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,
#                                   1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S3,nodeIdx,componentIdx,
#                                   iron.BoundaryConditionsTypes.FIXED,0.0)
#        boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,
#                                   1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2_S3,nodeIdx,componentIdx,
#                                   iron.BoundaryConditionsTypes.FIXED,0.0)
#        boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,
#                                   1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1_S2_S3,nodeIdx,componentIdx,
#                                   iron.BoundaryConditionsTypes.FIXED,0.0)
#    boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,
#                               1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeIdx,2,
#                               iron.BoundaryConditionsTypes.FIXED,0.0)
#    boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,
#                               1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeIdx,1,
#                               iron.BoundaryConditionsTypes.FIXED,0.0)
#    boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,
#                               1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeIdx,2,
#                               iron.BoundaryConditionsTypes.FIXED,0.0)
#    boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,
#                               1,iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeIdx,3,
#                               iron.BoundaryConditionsTypes.FIXED,0.0)
       
solverEquations.BoundaryConditionsCreateFinish()


#=================================================================
# S o l v e    a n d    E x p o r t    D a t a
#=================================================================
derivativeVector=[0.0,0.0,0.0,0.0]
numberOfIterations = 2
for iteration in range (startIteration,startIteration+numberOfIterations+1):
    # Solve the problem
    print("Solving fitting problem, iteration: " + str(iteration))
    problem.Solve()
    # Normalise derivatives
    for nodeIdx in range(1,numberOfNodes+1):
      for derivativeIdx in [iron.GlobalDerivativeConstants.GLOBAL_DERIV_S1,
                            iron.GlobalDerivativeConstants.GLOBAL_DERIV_S2,
                            iron.GlobalDerivativeConstants.GLOBAL_DERIV_S3]:
          length=0.0
          for componentIdx in range(1,4):
              derivativeVector[componentIdx]=dependentField.ParameterSetGetNode(iron.FieldVariableTypes.U,
                                                                                iron.FieldParameterSetTypes.VALUES,
                                                                                1,derivativeIdx,nodeIdx,componentIdx)
              length=length + derivativeVector[componentIdx]*derivativeVector[componentIdx]
          length=math.sqrt(length)
          for componentIdx in range(1,4):
              value=derivativeVector[componentIdx]/length
              dependentField.ParameterSetUpdateNode(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                    1,derivativeIdx,nodeIdx,componentIdx,value)
    # Copy dependent field to geometric 
    for componentIdx in range(1,numberOfDimensions+1):
        dependentField.ParametersToFieldParametersComponentCopy(iron.FieldVariableTypes.U,
                                                                iron.FieldParameterSetTypes.VALUES,
                                                                componentIdx,geometricField,
                                                                iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,
                                                                componentIdx)
    # Reproject
    dataProjection.DataPointsProjectionEvaluate(iron.FieldParameterSetTypes.VALUES)
    rmsError=dataProjection.ResultRMSErrorGet()
    print("RMS error = "+ str(rmsError))
    # Export fields
    print("Writing deformed geometry")
    fields = iron.Fields()
    fields.CreateRegion(region)
    fields.NodesExport("DeformedGeometry" + str(iteration),"FORTRAN")
    fields.ElementsExport("DeformedGeometry" + str(iteration),"FORTRAN")
    fields.Finalise()
#-----------------------------------------------------------------


iron.Finalise()
