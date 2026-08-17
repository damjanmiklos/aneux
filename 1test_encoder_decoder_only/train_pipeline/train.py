import torch
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, loss_fn, weights, device, accum_steps=1):
    model.train()
    
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_geom = 0.0
    total_samples = 0
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch = batch.to(device)
        batch_size = batch.num_graphs
        total_samples += batch_size
        
        # Forward pass with mixed precision bfloat16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x_pred, mu, logvar = model(batch)
            
            # Calculate losses — faces come per-sample from the Data object
            loss_recon, loss_kl, loss_geom = loss_fn(
                x_pred, 
                batch.x_true, 
                mu, 
                logvar, 
                batch.x, 
                batch.edge_index, 
                batch.x_true_batch, 
                batch.num_graphs,
                batch.faces if hasattr(batch, 'faces') and batch.faces is not None else None
            )
            
            # Weighted sum, scaled by accumulation steps
            loss = (weights['recon'] * loss_recon + weights['kl'] * loss_kl + weights['geom'] * loss_geom) / accum_steps
        
        # Backward pass (gradients accumulate)
        loss.backward()
        
        # Step optimizer every accum_steps, or at the last batch
        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
        
        # Track metrics (undo the /accum_steps scaling for logging)
        total_loss += loss.item() * accum_steps * batch_size
        total_recon += loss_recon.item() * batch_size
        total_kl += loss_kl.item() * batch_size
        total_geom += loss_geom.item() * batch_size
        
    return {
        'loss': total_loss / total_samples,
        'recon': total_recon / total_samples,
        'kl': total_kl / total_samples,
        'geom': total_geom / total_samples
    }

def evaluate_epoch(model, dataloader, loss_fn, weights, device):
    model.eval()
    
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_geom = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = batch.to(device)
            batch_size = batch.num_graphs
            total_samples += batch_size
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                x_pred, mu, logvar = model(batch)
                
                loss_recon, loss_kl, loss_geom = loss_fn(
                    x_pred, batch.x_true, mu, logvar, batch.x, batch.edge_index, batch.x_true_batch, batch.num_graphs,
                    batch.faces if hasattr(batch, 'faces') and batch.faces is not None else None
                )
                
                loss = weights['recon'] * loss_recon + weights['kl'] * loss_kl + weights['geom'] * loss_geom
            
            total_loss += loss.item() * batch_size
            total_recon += loss_recon.item() * batch_size
            total_kl += loss_kl.item() * batch_size
            total_geom += loss_geom.item() * batch_size
            
    return {
        'loss': total_loss / total_samples,
        'recon': total_recon / total_samples,
        'kl': total_kl / total_samples,
        'geom': total_geom / total_samples
    }

def train_model(model, train_dataset, val_dataset, epochs=100, batch_size=4, lr=1e-4, weights=None, device='cuda', accum_steps=1, val_every=5, num_workers=0):
    if weights is None:
        weights = {'recon': 1.0, 'kl': 0.001, 'geom': 0.1}
        
    # PyG DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        follow_batch=['x_true'], 
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, follow_batch=['x_true'], num_workers=0)
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    from losses import compute_losses
    
    print(f"Effective batch size: {batch_size} x {accum_steps} = {batch_size * accum_steps}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Faces are now per-sample in the Data object, no need for shared faces
    
    history = []
    
    for epoch in range(1, epochs + 1):
        # Training
        metrics = train_epoch(model, train_loader, optimizer, compute_losses, weights, device, accum_steps)
        
        print(f"Epoch {epoch:03d}/{epochs:03d} [TRAIN] | "
              f"Total: {metrics['loss']:.4f} | "
              f"Recon: {metrics['recon']:.4f} | "
              f"KL: {metrics['kl']:.4f} | "
              f"Geom: {metrics['geom']:.4f} | "
              f"weighed recon: {metrics['recon']*weights['recon']:.4f} | "
              f"weighed kl: {metrics['kl']*weights['kl']:.4f} | "
              f"weighed geom: {metrics['geom']*weights['geom']:.4f}") 
              
        # Validation
        if epoch % val_every == 0 or epoch == epochs:
            val_metrics = evaluate_epoch(model, val_loader, compute_losses, weights, device)
            print(f"Epoch {epoch:03d}/{epochs:03d} [VAL]   | "
                  f"Total: {val_metrics['loss']:.4f} | "
                  f"Recon: {val_metrics['recon']:.4f} | "
                  f"KL: {val_metrics['kl']:.4f} | "
                  f"Geom: {val_metrics['geom']:.4f} | "
                  f"weighed recon: {val_metrics['recon']*weights['recon']:.4f} | "
                  f"weighed kl: {val_metrics['kl']*weights['kl']:.4f} | "
                  f"weighed geom: {val_metrics['geom']*weights['geom']:.4f}") 
            metrics['val_loss'] = val_metrics['loss']
            
        history.append(metrics)
        
    return model, history
