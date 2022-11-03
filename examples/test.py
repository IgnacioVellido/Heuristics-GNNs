import copy
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils import *
from models import FRGCN, RACGNN

# Get data
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.loader import DataLoader


from torch.utils.tensorboard import SummaryWriter

# Set numpy seed
SEED = 12345
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------------------
# Get data
# ------------------------------------------------------------------------------

from_atlas = True

if from_atlas:
    Atlas = nx.graph_atlas_g()[3:]  # Graphs with more than 3 nodes

    # Get only fully connected
    Atlas_fc = [a for a in Atlas if nx.number_connected_components(a) == 1]

    # Select number of graphs
    N_GRAPHS = len(Atlas_fc) // 2

    idx = np.random.choice(len(Atlas_fc), N_GRAPHS, replace=False)

    graphs = [Atlas_fc[i] for i in idx]

else:
    N_GRAPHS = 1000
    MIN_NODES = 4
    MAX_NODES = 30

    # Get random graphs
    graphs = []
    for i in range(N_GRAPHS):
        # Generate fully connected random graph with and 0.0001 prob of an additional edge
        graphs.append(gnp_random_connected_graph(
            np.random.choice(np.arange(MIN_NODES, MAX_NODES)), 0.0001)
        ) #, 0.1))


# ------------------------------------------------------------------------------
# To dataloader
# ------------------------------------------------------------------------------

max_ecc = preprocess_graphs(graphs) # in-place modification

# Transform each graph to Data tensors
data_list = [from_networkx(g, group_node_attrs='x') for g in graphs]

# # Normalize
# y_list = normalize_labels(data_list,
#                         #   max=None) # Normalize each graph independently
#                           max=max_ecc) #, norm="other")

# # Return the values to the data list
# for y, data in zip(y_list, data_list):
#     data["y"] = y


# Train-test split
TRAIN_TEST_SPLIT = 0.8 # 0.1
TRAIN_SIZE = int(TRAIN_TEST_SPLIT * len(data_list))
TEST_SIZE = len(data_list) - TRAIN_SIZE

train_dataset, test_dataset = torch.utils.data.random_split(data_list, [TRAIN_SIZE, TEST_SIZE])

# Define batch loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ------------------------------------------------------------------------------
# Define GNN model
# ------------------------------------------------------------------------------

model = RACGNN(input_dim=1,
              hidden_dim=1,
              output_dim=1,
              aggregate_type="min",
              combine_type="mlp", #"mlp","simple"
              combine_layers=0,
              num_mlp_layers=3,
            #   task="node"
              final_mlp=False,
        ).to(DEVICE)

# model = FRGCN(in_channels=1, out_channels=1).to(DEVICE)

# ------------------------------------------------------------------------------
# Initialize optimizer
LR = 0.01
DECAY = 0.00001

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)
loss_fn = torch.nn.L1Loss()

# ------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------

ITERATIONS = 500
PATIENCE = 30
BATCH_SIZE = 1


tag = "train"
model.train()

# To calculate metrics
losses = []
best_batch_loss = 10000
best_model = model

similarities = []
best_batch_similarity = 0

best_iter = 0

writer = SummaryWriter(f'runs/setting_up')

# ------------------------------------------------------------------------------
no_improvements = 0
iter = 0

# LOOP
while iter < ITERATIONS and no_improvements < PATIENCE:
    iter += 1

    for i, data in enumerate(train_loader):
        data = data.to(DEVICE)

        # Forward
        ecc = data['eccentricity'].item()
        passes = ecc + np.random.choice(np.arange(1, ecc+1))

        out = model(data['x'], data['edge_index'], data['batch'], passes=passes)

        out = out.T.squeeze()

        # Backpropagation
        loss = loss_fn(out, data['y']) / BATCH_SIZE
        loss.backward()

        # Save metrics
        losses.append(loss.item())

        sim, _ = graphs_similarity(data['y'], out, decimals=0)
        similarities.append(sim)


        # Optimize only after batch completed
        if (i+1) % BATCH_SIZE == 0:
            optimizer.step()

            # Reset gradients for the next batch
            optimizer.zero_grad()


    # --------------------------------------------------------------------------
    # Keep best model
    batch_loss = np.mean(losses[-TRAIN_SIZE:-1]) * BATCH_SIZE
    batch_similarity = np.mean(similarities[-TRAIN_SIZE:-1])

    if (best_batch_loss - batch_loss) > 0.005: # 0.001 for shortestPathLoss
        print(f"[Iter: {iter}] Saving best model - Loss: {batch_loss}\tAvg-similarity: {batch_similarity}")
        
        best_batch_loss = batch_loss
        best_model = copy.deepcopy(model)
        best_batch_similarity = batch_loss
        best_iter = iter

        no_improvements = 0
    else:
        no_improvements += 1

    # --------------------------------------------------------------------------
    # Write to Tensorboard
    writer.add_scalar(f'loss/{tag}', batch_loss, iter)
    writer.add_scalar(f'avg_similarity/{tag}', batch_similarity, iter)

    # --------------------------------------------------------------------------
    # Print
    if iter % 10 == 0:
        print(f"[Iter: {iter}] Batch-Loss: {batch_loss}\tAvg-similarity: {batch_similarity}")


# ------------------------------------------------------------------------------
writer.add_hparams({'lr': LR, 'decay': DECAY, 'iterations': best_iter},
                   {'loss': best_batch_loss, 'avg_similarity': best_batch_similarity})


# ------------------------------------------------------------------------------
# Test
# ------------------------------------------------------------------------------

def test(loader, model, extra_passes=3, decimals=0, plot=True):
    model.eval()

    # To calculate metrics
    losses = []
    similarities = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            
            # Forward
            passes = batch['eccentricity'].item() + extra_passes

            out = model(batch['x'], batch['edge_index'], batch['batch'], passes=passes)

            out = out.T.squeeze()

            # Loss
            loss = loss_fn(out, batch.y)

            # ------------------------------------------------------------------
            # Log and print
            losses.append(loss.item())

            sim, rounded = graphs_similarity(batch.y, out, decimals)
            similarities.append(sim)

            # ------------------------------------------------------------------
            # Plot graphs (more blue = nearer to goal)
            if plot:
                # Print stats
                print(f"Passes: {passes}")
                print(out.T.squeeze())
                print(rounded)
                print(batch.y)
                print(f"Loss: {loss.item()}\tSimilarity: {sim}")


                # --------------------------------------------------------------
                # Get graph
                G_out = to_networkx(batch, node_attrs=['x', 'y'], edge_attrs=['weight'],
                                    to_undirected=True, remove_self_loops=True)    # NOTE: Careful with self loops

                fig, ax = plt.subplots(1,2, figsize=(17,7))

                # Get pos as a circular layout
                pos = nx.circular_layout(G_out)

                # Plot edge labels
                edge_labels = nx.get_edge_attributes(G_out,'weight')
                nx.draw_networkx_edge_labels(G_out, pos=pos, edge_labels=edge_labels)

                # Plots sorted, with lower value being blue
                # color_order = np.argsort(np.argsort(out.T.squeeze()))
                # color_order = out.T.squeeze()

                ax[0].set_title("Learnt")
                pred = out.cpu().detach() if torch.cuda.is_available() else out
                nx.draw(G_out, pos=pos, cmap=plt.get_cmap('coolwarm'), node_color=pred, 
                        ax=ax[0],
                        with_labels=True, font_color='black')

                ax[1].set_title("True")
                true = batch['y'].cpu().detach() if torch.cuda.is_available() else batch['y']
                nx.draw(G_out, pos=pos, cmap=plt.get_cmap('coolwarm'), node_color=true,
                        ax=ax[1],
                        with_labels=True, font_color='black')
                
                plt.show()


    print(f"Avg loss {np.mean(losses)}")
    print(f"Std loss {np.std(losses)}")
    print(f"Avg similarity {np.mean(similarities)}")
    print(f"Std similarity {np.std(similarities)}")



# TRAIN
test(train_loader, best_model, extra_passes=50, decimals=0, plot=False)


# TEST
test(test_loader, best_model, extra_passes=50, decimals=0, plot=False)


# ------------------------------------------------------------------------------
# Test with various goals

batch = data_list[0].to(DEVICE)

x = batch['x'].clone()

# Add one goal
x[5] = 1.0

print(x.T)
print(batch.y)
print("\n\n")

out = best_model(x, batch['edge_index'], batch['weight'], passes=100)
print(f"Out: {out.T}")

sim, pred = graphs_similarity(batch.y, out, decimals=0)
print(f"Rounded: {pred}")
print(f"Similarity: {sim}")

# ------------------------------------------------------------------
# Plot graphs (more blue = nearer to goal)

# Get graph
G_out = to_networkx(batch, node_attrs=['x', 'y'],
                    to_undirected=True, remove_self_loops=True)    # NOTE: Careful with self loops

fig, ax = plt.subplots(1,2, figsize=(17,7))

# Plots sorted, with lower value being blue
# color_order = np.argsort(np.argsort(out.T.squeeze()))
# color_order = out.T.squeeze()

ax[0].set_title("Learnt")
# pred = out.cpu().detach() if torch.cuda.is_available() else out
nx.draw_circular(G_out, cmap=plt.get_cmap('coolwarm'), node_color=pred, 
                ax=ax[0],
                with_labels=True, font_color='black')

ax[1].set_title("True")
true = batch['y'].cpu().detach() if torch.cuda.is_available() else batch['y']
nx.draw_circular(G_out, cmap=plt.get_cmap('coolwarm'), node_color=true,
                ax=ax[1],
                with_labels=True, font_color='black')

plt.show()