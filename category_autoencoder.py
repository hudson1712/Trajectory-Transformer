import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define categorical variables
categories = [3, 3, 4, 2]  # Number of possible values for each category

# Calculate input dimension
input_dim = sum(categories)

# Define hyperparameters
code_dim = 4
learning_rate = 0.001
num_epochs = 100
batch_size = 32

# Define autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, code_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, code_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim),
            nn.Sigmoid()  # Sigmoid activation to constrain output between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Create random categorical data
def generate_data(num_samples, categories):
    data = []
    for _ in range(num_samples):
        sample = []
        for num_values in categories:
            category_index = np.random.randint(0, num_values)
            category_vector = [0] * num_values
            category_vector[category_index] = 1
            sample.extend(category_vector)
        data.append(sample)
    return np.array(data)

# Convert data to PyTorch tensors
def to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)

# Generate random categorical data
num_samples = 1000
data = generate_data(num_samples, categories)

# Convert data to PyTorch tensors
input_data = to_tensor(data)

# Initialize the autoencoder model
model = Autoencoder(input_dim, code_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Shuffle data
    indices = torch.randperm(input_data.size(0))
    shuffled_data = input_data[indices]
    
    # Mini-batch training
    for i in range(0, input_data.size(0), batch_size):
        # Get mini-batch
        batch_data = shuffled_data[i:i+batch_size]
        
        # Forward pass
        output = model(batch_data)
        loss = criterion(output, batch_data)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print epoch loss
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Convert output to numpy array
output_data = model(input_data).detach().numpy()

# Print original and reconstructed data for comparison
print("\nOriginal data:")
print(data[:5])
print("\nReconstructed data:")
print(output_data[:5])

def get_encodings(model, input_data):
    with torch.no_grad():
        encoded_data = model.encoder(input_data)
    return encoded_data.numpy()

# Get encodings for input data
encodings = get_encodings(model, input_data)

# Print encodings
print("Encoded data:")
print(encodings)