#!/usr/bin/env python
# coding: utf-8

# Modularity on the Karate Club Graph

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.linalg.modularitymatrix import modularity_matrix


# In[2]:


# use a function to do recursive bisection to detect multiple communities
# graph should be drawn using a fixed spring layout
# assign a unique colour to each community
# compute node metrics after each split
# procedure: we have a graph G(V,E). Make a graph with 34 nodes and 78 edges
# make sure it is a connected graph?
# import networkx and build the karate club graph
# we start with the full vertex set V with all nodes
# calculate the modularity matrix B for all the nodes
# find the leading eigenvector and eigenvalue(largest)
# Find the leading eigenvector: the one associated with the largest eigenvalue.
# Check if this eigenvalue is positive (indicates a modularity gain is possible).
# For each node, look at the corresponding value in the leading eigenvector:
# If it's positive, assign the node to Group 1.
# If it's negative, assign the node to Group 2.
# now we have 2 communities, each is an induced subgraph
# now repeat this process for each subgraph
# if the leading eigenvalue(largest) is negative/ the group is too small(3). stop splitting


# In[3]:


import networkx as nx
G = nx.karate_club_graph()
pos = nx.spring_layout(G, seed = 42)
nx.draw(G, pos, with_labels=True, node_color='skyblue', font_weight='bold', node_size=500)


# In[4]:


# Modularity matrix for the entire set of nodes
# B = A - k(i)k(j)/2m
vertex_set = list(G.nodes())
print(vertex_set)


# In[5]:


for node in list(G.nodes()):
    #to compute the adjacency matrix
    adjacency_matrix = list(G.edges())
    #print(adjacency_matrix)


# In[6]:


# adjacency matrix
n = len(list(G.nodes()))
nodes = list(G.nodes())
A = np.zeros((n ,n))
for i in range(n):
    for j in range(n):
        if (nodes[i], nodes[j]) in list(G.edges()) or (nodes[j], nodes[i]) in list(G.edges()): 
            A[i,j] = 1 
print(A)


# In[7]:


# A1 = nx.to_numpy_array(G, dtype=int)
# print(A1)


# In[8]:


# Number of edges
m = G.number_of_edges()

# Degree vector k
k = np.array([degree for node, degree in G.degree()])

# Outer product kk^T
kkT = np.outer(k, k)

# Compute kk^T / (2m)
kkT_div = kkT / (2 * m)

print("Degree vector k:", k)
print("Matrix kk^T / 2m:\n", kkT_div)


# In[9]:


# modularity matrix
B = A - kkT_div
print(B)


# In[10]:


# finding the leading eigenvector and eigenvalue of the modularity matrix for the full node set V
# for i,j in B[i,j]:
    # if i=j
eigenvalues, eigenvectors = np.linalg.eig(B)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors: ", eigenvectors)

max_index = np.argmax(eigenvalues)
max_eigenvector = eigenvectors[:, max_index]
    
eigenvalues = np.sort(eigenvalues)[::-1]  # descending order
print("Max eigenvalue: ", max(eigenvalues))
print("Corresponding Eigenvector:\n", max_eigenvector)


# In[11]:


# splitting the graph for the 1st time
group1 = []
group2 = []
for value in max_eigenvector:
    if value < 0:
        group1.append(value)
    else:
        group2.append(value)
print("Group 1: ", group1)
print("Group 2: ", group2)


# In[12]:


group1 = [nodes[i] for i, val in enumerate(max_eigenvector) if val < 0]
group2 = [nodes[i] for i, val in enumerate(max_eigenvector) if val >= 0]

print("Group 1 Nodes:", group1)
print("Group 2 Nodes:", group2)


# In[13]:


# computing node metrics after first split

# Create subgraphs for each group
G1 = G.subgraph(group1)
G2 = G.subgraph(group2)

# Compute node metrics for Group 1
print("Group 1 Node degrees:")
print(dict(G1.degree()))

print("Group 1 Clustering coefficient:")
print(nx.clustering(G1))

# Compute node metrics for Group 2
print("Group 2 Node degrees:")
print(dict(G2.degree()))

print("Group 2 Clustering coefficient:")
print(nx.clustering(G2))


# In[14]:


G1 = G.subgraph(group1)
pos = nx.spring_layout(G, seed = 42)
nx.draw(G1, pos, with_labels=True, node_color='purple', font_weight='bold', node_size=500)

G2 = G.subgraph(group2)
pos = nx.spring_layout(G, seed = 42)
nx.draw(G2, pos, with_labels=True, node_color='darkblue', font_weight='bold', node_size=500)


# In[15]:


n1 = len(list(G1.nodes()))
nodes1 = list(G1.nodes())
A1 = np.zeros((n1 ,n1))
for i in range(n1):
    for j in range(n1):
        if (nodes1[i], nodes1[j]) in list(G1.edges()) or (nodes1[j], nodes1[i]) in list(G1.edges()): 
            A1[i,j] = 1 

# Number of edges
m1 = G1.number_of_edges()

# Degree vector k
k1 = np.array([degree for node, degree in G1.degree()])

# Outer product kk^T
kkT1 = np.outer(k1, k1)

# Compute kk^T / (2m)
kkT_div1 = kkT1 / (2 * m1)

# modularity matrix
B1 = A1 - kkT_div1

eigenvalues, eigenvectors = np.linalg.eig(B1)

max_index = np.argmax(eigenvalues)
max_eigenvector = eigenvectors[:, max_index]
    
eigenvalues = np.sort(eigenvalues)[::-1]  # descending order
print("Max eigenvalue: ", max(eigenvalues))

# splitting the subgraph for the 1st time
group1 = []
group2 = []
for value in max_eigenvector:
    if value < 0:
        group1.append(value)
    else:
        group2.append(value)

group1 = [nodes1[i] for i, val in enumerate(max_eigenvector) if val < 0]
group2 = [nodes1[i] for i, val in enumerate(max_eigenvector) if val >= 0]

print("Group 1 Nodes:", group1)
print("Group 2 Nodes:", group2)

pos = nx.spring_layout(G, seed=42)

G3 = G1.subgraph(group1)
pos_G3 = {node: pos[node] for node in G3.nodes()}
nx.draw(G3, pos_G3, with_labels=True, node_color='pink', font_weight='bold', node_size=500)

G4 = G1.subgraph(group2)
pos_G4 = {node: pos[node] for node in G4.nodes()}
nx.draw(G4, pos_G4, with_labels=True, node_color='cyan', font_weight='bold', node_size=500)


n2 = len(list(G2.nodes()))
nodes2 = list(G2.nodes())
A2 = np.zeros((n2 ,n2))
for i in range(n2):
    for j in range(n2):
        if (nodes2[i], nodes2[j]) in list(G2.edges()) or (nodes2[j], nodes2[i]) in list(G2.edges()): 
            A2[i,j] = 1 

# Number of edges
m2 = G2.number_of_edges()

# Degree vector k
k2 = np.array([degree for node, degree in G2.degree()])

# Outer product kk^T
kkT2 = np.outer(k2, k2)

# Compute kk^T / (2m)
kkT_div2 = kkT2 / (2 * m2)

# modularity matrix
B2 = A2 - kkT_div2

eigenvalues, eigenvectors = np.linalg.eig(B2)

max_index = np.argmax(eigenvalues)
max_eigenvector = eigenvectors[:, max_index]
    
eigenvalues = np.sort(eigenvalues)[::-1]  # descending order
print("Max eigenvalue: ", max(eigenvalues))

# splitting the subgraph for the 2nd time
group1 = []
group2 = []
for value in max_eigenvector:
    if value < 0:
        group1.append(value)
    else:
        group2.append(value)

group1 = [nodes2[i] for i, val in enumerate(max_eigenvector) if val < 0]
group2 = [nodes2[i] for i, val in enumerate(max_eigenvector) if val >= 0]

print("Group 1 Nodes:", group1)
print("Group 2 Nodes:", group2)

pos = nx.spring_layout(G, seed=42)

G5 = G2.subgraph(group1)
pos_G5 = {node: pos[node] for node in G5.nodes()}
nx.draw(G5, pos_G5, with_labels=True, node_color='red', font_weight='bold', node_size=500)

G6 = G2.subgraph(group2)
pos_G6 = {node: pos[node] for node in G6.nodes()}
nx.draw(G6, pos_G6, with_labels=True, node_color='green', font_weight='bold', node_size=500)


# In[16]:


# computing node metrics after second split

# Create subgraphs for each group
G5 = G2.subgraph(group1)
G6 = G2.subgraph(group2)

# Compute node metrics for Group 1
print("Group 1 Node degrees:")
print(dict(G5.degree()))

print("Group 1 Clustering coefficient:")
print(nx.clustering(G5))

# Compute node metrics for Group 2
print("Group 2 Node degrees:")
print(dict(G6.degree()))

print("Group 2 Clustering coefficient:")
print(nx.clustering(G6))

print(nx.degree_centrality(G5))
print(nx.betweenness_centrality(G5))
print(nx.closeness_centrality(G5))

print(nx.degree_centrality(G6))
print(nx.betweenness_centrality(G6))
print(nx.closeness_centrality(G6))


# In[17]:


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Assume G is the main Karate Club graph and G3, G4, G5, G6 are defined subgraphs
G = nx.karate_club_graph()
pos = nx.spring_layout(G, seed=42)  # fixed positions for all subgraphs

# ========== SUBGRAPH G3 ==========
n3 = len(list(G3.nodes()))
if n3 < 2 or G3.number_of_edges() == 0:
    print("Subgraph G3 too small or no edges; stop splitting")
else:
    nodes3 = list(G3.nodes())
    A3 = np.zeros((n3, n3))
    for i in range(n3):
        for j in range(n3):
            if (nodes3[i], nodes3[j]) in list(G3.edges()) or (nodes3[j], nodes3[i]) in list(G3.edges()):
                A3[i, j] = 1

    m3 = G3.number_of_edges()
    k3 = np.array([degree for node, degree in G3.degree()])
    kkT3 = np.outer(k3, k3)
    B3 = A3 - kkT3 / (2 * m3)

    eigenvalues3, eigenvectors3 = np.linalg.eig(B3)
    max_index3 = np.argmax(eigenvalues3.real)
    max_eigenvector3 = eigenvectors3[:, max_index3].real

    group1 = [nodes3[i] for i, val in enumerate(max_eigenvector3) if val < 0]
    group2 = [nodes3[i] for i, val in enumerate(max_eigenvector3) if val >= 0]

    print("\nG3 Splitting:")
    print("Max eigenvalue:", max(eigenvalues3.real))
    print("Group 1 Nodes:", group1)
    print("Group 2 Nodes:", group2)

    G7 = G3.subgraph(group1)
    G8 = G3.subgraph(group2)
    nx.draw(G7, pos, with_labels=True, node_color='grey', font_weight='bold', node_size=500)
    nx.draw(G8, pos, with_labels=True, node_color='yellow', font_weight='bold', node_size=500)


# ========== SUBGRAPH G4 ==========
n4 = len(list(G4.nodes()))
if n4 < 2 or G4.number_of_edges() == 0:
    print("Subgraph G4 too small or no edges; stop splitting")
else:
    nodes4 = list(G4.nodes())
    A4 = np.zeros((n4, n4))
    for i in range(n4):
        for j in range(n4):
            if (nodes4[i], nodes4[j]) in list(G4.edges()) or (nodes4[j], nodes4[i]) in list(G4.edges()):
                A4[i, j] = 1

    m4 = G4.number_of_edges()
    k4 = np.array([degree for node, degree in G4.degree()])
    kkT4 = np.outer(k4, k4)
    B4 = A4 - kkT4 / (2 * m4)

    eigenvalues4, eigenvectors4 = np.linalg.eig(B4)
    max_index4 = np.argmax(eigenvalues4.real)
    max_eigenvector4 = eigenvectors4[:, max_index4].real

    group1 = [nodes4[i] for i, val in enumerate(max_eigenvector4) if val < 0]
    group2 = [nodes4[i] for i, val in enumerate(max_eigenvector4) if val >= 0]

    print("\nG4 Splitting:")
    print("Max eigenvalue:", max(eigenvalues4.real))
    print("Group 1 Nodes:", group1)
    print("Group 2 Nodes:", group2)

    G9 = G4.subgraph(group1)
    G10 = G4.subgraph(group2)
    nx.draw(G9, pos, with_labels=True, node_color='red', font_weight='bold', node_size=500)
    nx.draw(G10, pos, with_labels=True, node_color='green', font_weight='bold', node_size=500)


# ========== SUBGRAPH G5 ==========
n5 = len(list(G5.nodes()))
if n5 < 2 or G5.number_of_edges() == 0:
    print("Subgraph G5 too small or no edges; stop splitting")
else:
    nodes5 = list(G5.nodes())
    A5 = np.zeros((n5, n5))
    for i in range(n5):
        for j in range(n5):
            if (nodes5[i], nodes5[j]) in list(G5.edges()) or (nodes5[j], nodes5[i]) in list(G5.edges()):
                A5[i, j] = 1

    m5 = G5.number_of_edges()
    k5 = np.array([degree for node, degree in G5.degree()])
    kkT5 = np.outer(k5, k5)
    B5 = A5 - kkT5 / (2 * m5)

    eigenvalues5, eigenvectors5 = np.linalg.eig(B5)
    max_index5 = np.argmax(eigenvalues5.real)
    max_eigenvector5 = eigenvectors5[:, max_index5].real

    group1 = [nodes5[i] for i, val in enumerate(max_eigenvector5) if val < 0]
    group2 = [nodes5[i] for i, val in enumerate(max_eigenvector5) if val >= 0]

    print("\nG5 Splitting:")
    print("Max eigenvalue:", max(eigenvalues5.real))
    print("Group 1 Nodes:", group1)
    print("Group 2 Nodes:", group2)

    G11 = G5.subgraph(group1)
    G12 = G5.subgraph(group2)
    nx.draw(G11, pos, with_labels=True, node_color='pink', font_weight='bold', node_size=500)
    nx.draw(G12, pos, with_labels=True, node_color='cyan', font_weight='bold', node_size=500)


# ========== SUBGRAPH G6 ==========
n6 = len(list(G6.nodes()))
if n6 < 2 or G6.number_of_edges() == 0:
    print("Subgraph G6 too small or no edges; stop splitting")
else:
    nodes6 = list(G6.nodes())
    A6 = np.zeros((n6, n6))
    for i in range(n6):
        for j in range(n6):
            if (nodes6[i], nodes6[j]) in list(G6.edges()) or (nodes6[j], nodes6[i]) in list(G6.edges()):
                A6[i, j] = 1

    m6 = G6.number_of_edges()
    k6 = np.array([degree for node, degree in G6.degree()])
    kkT6 = np.outer(k6, k6)
    B6 = A6 - kkT6 / (2 * m6)

    eigenvalues6, eigenvectors6 = np.linalg.eig(B6)
    max_index6 = np.argmax(eigenvalues6.real)
    max_eigenvector6 = eigenvectors6[:, max_index6].real

    group1 = [nodes6[i] for i, val in enumerate(max_eigenvector6) if val < 0]
    group2 = [nodes6[i] for i, val in enumerate(max_eigenvector6) if val >= 0]

    print("\nG6 Splitting:")
    print("Max eigenvalue:", max(eigenvalues6.real))
    print("Group 1 Nodes:", group1)
    print("Group 2 Nodes:", group2)

    G13 = G6.subgraph(group1)
    G14 = G6.subgraph(group2)
    nx.draw(G13, pos, with_labels=True, node_color='indigo', font_weight='bold', node_size=500)
    nx.draw(G14, pos, with_labels=True, node_color='violet', font_weight='bold', node_size=500)

# Show all splits on one graph
plt.title("Karate Club Graph - All Subgraph Splits")
plt.axis('off')
plt.show()


# In[18]:


# List of subgraphs from G7 to G14
subgraphs = [G7, G8, G9, G10, G11, G12, G13, G14]

# Compute and print metrics for each subgraph
for i, subgraph in enumerate(subgraphs, start=7):
    print(f"\nGroup {i} Node degrees:")
    print(dict(subgraph.degree()))
    
    print(f"Group {i} Clustering coefficient:")
    print(nx.clustering(subgraph))
    
    print(f"Group {i} Degree centrality:")
    print(nx.degree_centrality(subgraph))
    
    print(f"Group {i} Betweenness centrality:")
    print(nx.betweenness_centrality(subgraph))
    
    print(f"Group {i} Closeness centrality:")
    print(nx.closeness_centrality(subgraph))


# In[27]:


import networkx as nx

# List of all subgraphs from G1 to G14
all_subgraphs = [G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13, G14]

# Initialize storage for metrics
metrics_over_time = []

# Compute and store metrics for each subgraph
for subgraph in all_subgraphs:
    if subgraph.number_of_nodes() > 1:
        metrics = {
            'degree': nx.degree_centrality(subgraph),
            'betweenness': nx.betweenness_centrality(subgraph),
            'closeness': nx.closeness_centrality(subgraph),
            'clustering': nx.clustering(subgraph)
        }
        metrics_over_time.append(metrics)
#print(metrics_over_time)


# import matplotlib.pyplot as plt
# import numpy as np
# 
# # Plot the evolution of metrics
# node_ids = sorted(G.nodes())
# num_splits = len(metrics_over_time)
# 
# for metric_name in ['degree', 'betweenness', 'closeness', 'clustering']:
#     plt.figure(figsize=(12, 6))
#     has_lines = False  # Track if any lines are plotted
#     for node in node_ids:
#         values = []
#         for metrics in metrics_over_time:
#             value = metrics[metric_name].get(node, None)
#             values.append(value)
#         values = np.array([v if v is not None else np.nan for v in values])
#         
#         # Find segments of consecutive non-nan values
#         segments = []
#         start = None
#         for i, val in enumerate(values):
#             if not np.isnan(val):
#                 if start is None:
#                     start = i
#             else:
#                 if start is not None:
#                     segments.append((start, i))
#                     start = None
#         if start is not None:
#             segments.append((start, len(values)))
#         
#         # Plot each segment
#         for start, end in segments:
#             if end - start > 1:  # Only plot if there are at least two points
#                 plt.plot(range(start, end), values[start:end], label=f'Node {node}', alpha=0.7)
#                 has_lines = True  # Mark that at least one line was plotted
#                 print(f"Plotting {metric_name} for Node {node}: {values[start:end]}")
#     
#     plt.xlabel('Split Iteration')
#     plt.ylabel(f'{metric_name.title()} Centrality')
#     plt.title(f'Evolution of {metric_name.title()} Centrality Across Splits')
#     
#     # Only call legend if there are labeled lines
#     if has_lines:
#         plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='x-small', ncol=2)
#     
#     plt.tight_layout()
#     plt.show()
# 

# CODE FOR A GENERAL GRAPH

# In[ ]:





# In[ ]:





# In[ ]:





# In[49]:


G = nx.erdos_renyi_graph(n=10, p=0.3)
pos = nx.spring_layout(G, seed = 42)
nx.draw(G, with_labels=True)
plt.show()


# In[50]:


n1 = len(list(G.nodes()))
nodes1 = list(G.nodes())
A1 = np.zeros((n1 ,n1))
for i in range(n1):
    for j in range(n1):
        if (nodes1[i], nodes1[j]) in list(G1.edges()) or (nodes1[j], nodes1[i]) in list(G.edges()): 
            A1[i,j] = 1 
print(A1)


# In[51]:


m1 = G.number_of_edges()
k1 = np.array([degree for node, degree in G.degree()])
kkT1 = np.outer(k1, k1)
# Compute kk^T / (2m)
kkT_div1 = kkT1 / (2 * m1)
print(kkT_div)


# In[52]:


# modularity matrix
B1 = A1 - kkT_div1
print(B1)


# In[53]:


eigenvalues, eigenvectors = np.linalg.eig(B1)
max_index = np.argmax(eigenvalues.real)
max_eigenvector = eigenvectors[:, max_index].real
eigenvalues = np.sort(eigenvalues)[::-1]  # descending order
print("Max eigenvalue: ", max(eigenvalues))


# In[54]:


group1 = []
group2 = []
for value in max_eigenvector:
    if value < 0:
        group1.append(value)
    else:
        group2.append(value)
            
#assigning eigenvector values to nodoes
group1 = [nodes1[i] for i, val in enumerate(max_eigenvector) if val < 0]
group2 = [nodes1[i] for i, val in enumerate(max_eigenvector) if val >= 0]

print("Group 1 Nodes:", group1)
print("Group 2 Nodes:", group2)


# In[55]:


G1 = G.subgraph(group1)
pos = nx.spring_layout(G, seed = 42)
nx.draw(G1, pos, with_labels=True, node_color='purple', font_weight='bold', node_size=500)

G2 = G.subgraph(group2)
pos = nx.spring_layout(G, seed = 42)
nx.draw(G2, pos, with_labels=True, node_color='darkblue', font_weight='bold', node_size=500)


# In[7]:


###### Code for any general graph ######




import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def recursive_split(G, pos=None, depth=0, max_depth=3): # depth and max depth helps in recursive splitting
    
    if len(G.nodes()) < 3 or G.number_of_edges() == 0 or depth >= max_depth:
        print(f" {'  '*depth} Subgraph is too small or no edges; stop splitting")
        return
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    n1 = len(list(G.nodes()))
    nodes1 = list(G.nodes())
    A1 = np.zeros((n1 ,n1))
    for i in range(n1):
        for j in range(n1):
            if (nodes1[i], nodes1[j]) in list(G.edges()) or (nodes1[j], nodes1[i]) in list(G.edges()): 
                    A1[i,j] = 1 
            
    m1 = G.number_of_edges()
    k1 = np.array([degree for node, degree in G.degree()])
    kkT1 = np.outer(k1, k1)
    # Compute kk^T / (2m)
    kkT_div1 = kkT1 / (2 * m1)

    
    # modularity matrix
    B1 = A1 - kkT_div1
    
    eigenvalues, eigenvectors = np.linalg.eig(B1)
    max_index = np.argmax(eigenvalues.real)
    max_eigenvector = eigenvectors[:, max_index].real
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending order
    print(f"{'  '*depth}Max eigenvalue: ", max(eigenvalues))
    
        
    # splitting the subgraph for the 1st time
    group1 = []
    group2 = []
    for value in max_eigenvector:
        if value < 0:
            group1.append(value)
        else:
            group2.append(value)
        
    #assigning eigenvector values to nodes
    group1 = [nodes1[i] for i, val in enumerate(max_eigenvector) if val < 0]
    group2 = [nodes1[i] for i, val in enumerate(max_eigenvector) if val >= 0]

    if not group1 or not group2:
        print(f"{'  '*depth}Cannot split further.")
        return

    print(f"{'  '*depth}Group 1 Nodes:", group1)
    print(f"{'  '*depth}Group 2 Nodes:", group2)

    plt.figure(figsize=(6, 4))
    G_sub1 = G.subgraph(group1).copy()
    pos_G_sub1 = {node: pos[node] for node in G_sub1.nodes()}
    nx.draw(G_sub1, pos_G_sub1, with_labels=True, node_color='pink', font_weight='bold', node_size=500)
    plt.title(f"Depth {depth}")
    plt.show()
    
    plt.figure(figsize=(6, 4))
    G_sub2 = G.subgraph(group2).copy()
    pos_G_sub2 = {node: pos[node] for node in G_sub2.nodes()}
    nx.draw(G_sub2, pos_G_sub2, with_labels=True, node_color='cyan', font_weight='bold', node_size=500)
    plt.title(f"Depth {depth}")
    plt.show()

    # computing node metrics after each split


    # Compute node metrics for Group 1
    print(f"{'  '*depth}Group 1 Node degrees:")
    print(dict(G_sub1.degree()))

    print(f"{'  '*depth}Group 1 Clustering coefficient:")
    print(nx.clustering(G_sub1))

    # Compute node metrics for Group 2
    print(f"{'  '*depth}Group 2 Node degrees:")
    print(dict(G_sub2.degree()))

    print(f"{'  '*depth}Group 2 Clustering coefficient:")
    print(nx.clustering(G_sub2))

    print(nx.degree_centrality(G_sub1))
    print(nx.betweenness_centrality(G_sub1))
    print(nx.closeness_centrality(G_sub1))
    
    print(nx.degree_centrality(G_sub2))
    print(nx.betweenness_centrality(G_sub2))
    print(nx.closeness_centrality(G_sub2))
    
    recursive_split(G_sub1, pos, depth=depth+1, max_depth=max_depth)
    recursive_split(G_sub2, pos, depth=depth+1, max_depth=max_depth)


G = nx.erdos_renyi_graph(n = 25,p = 0.35, seed = 42) # N = number of nodes, P = probability of generating an edge
recursive_split(G)


# In[ ]:





# In[ ]:





# Short Discussion: Central Nodes and Community Structure Impact
# Throughout recursive spectral modularity partitioning, certain nodes in the karate club graph, particularly node 0 (“Mr. Hi”) and node 33 (the club president), consistently exhibit high centrality across degree, betweenness, and closeness metrics. These nodes frequently act as bridges and hubs, reflecting their pivotal roles in holding communities together and in facilitating cross-community connections. Because the spectral cuts are based on optimizing modularity, splits usually separate these leaders in the early stages, but their local centrality remains elevated even as communities subdivide further.
# 
# As community structure evolves with each split, centrality measures generally tend to decrease for most nodes, especially as large communities fragment into smaller, tightly knit groups. The clustering coefficient for most nodes tends to increase in later splits, reflecting the emergence of smaller, denser communities; however, nodes on the boundaries or bridges between communities may show irregular or even decreased clustering. Nodes that maintain high betweenness and closeness through multiple splits can be identified as crucial structural "brokers" and indicate persistent influence or connectivity irrespective of community granularity. Overall, recursive partitioning highlights the adaptation of each metric to community context: degree and betweenness centrality most clearly identify leaders and boundary nodes, while clustering coefficients illuminate cohesion within the emerging communities

# In[ ]:




