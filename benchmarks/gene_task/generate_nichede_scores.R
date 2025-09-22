# Set up system environment and install Niche-DE and other dependencies
Sys.setenv(R_REMOTES_NO_ERRORS_FROM_WARNINGS="true")
options(timeout=9999999)

IRkernel::installspec(user = TRUE)
devtools::install_github("amc-heme/sceasy")
devtools::install_github("kaishumason/NicheDE")

# Import dependencies
library(nicheDE)
library(sceasy)
library(reticulate)
library(Seurat)
library(anndata)
library(stringr)

print("Successfully installed dependencies...")

# Load the dataset
dataset_path <- snakemake@input[[1]]

seurat_dataset_path <- str_replace(dataset_path, ".h5ad", ".rds")

# Check if Seurat object already exists before converting
if (!file.exists(seurat_dataset_path)) {
    sceasy::convertFormat(dataset_path,
                         from="anndata",
                         to="seurat",
                         outFile=seurat_dataset_path,
                         main_layer = "counts")
}

# Load the Seurat object
seurat_obj <- readRDS(seurat_dataset_path)

# Create deconvolution matrix with cell type labels
cell_types <- seurat_obj$leiden
cell_names <- colnames(seurat_obj)
cell_type_mat <- matrix(0, nrow=length(cell_names), ncol=length(unique(cell_types)))
colnames(cell_type_mat) <- sort(unique(cell_types))
rownames(cell_type_mat) <- cell_names

# Fill the deconvolution matrix
for(i in 1:length(cell_names)) {
    cell_type_mat[i, cell_types[i]] <- 1
}

# Convert counts to integers by rounding
counts_matrix <- GetAssayData(seurat_obj, layer = "counts", assay = "RNA")
counts_matrix <- round(counts_matrix)
seurat_obj[["RNA"]]@counts <- counts_matrix

spatial_coords <- seurat_obj[["spatial"]]@cell.embeddings

# Create proper spatial data format for the SlideSeq image
spatial_data <- data.frame(
    imagerow = spatial_coords[,1],
    imagecol = spatial_coords[,2],
    row.names = colnames(seurat_obj)
)

seurat_obj@images <- list()  # Clear existing images
seurat_obj@images[[1]] <- new(
    Class = 'SlideSeq',
    coordinates = spatial_data,
    assay = "RNA",
    key = "image_"
)
names(seurat_obj@images) <- "coords"

# Set default assay to RNA
DefaultAssay(seurat_obj) <- "RNA"

# Calculate average expression profiles
cell_types <- unique(seurat_obj$leiden)
counts_matrix <- GetAssayData(seurat_obj, layer = "counts", assay = "RNA")

# Ensure cell types are consistently sorted
cell_types_sorted <- sort(unique(cell_types))

# Create library matrix with correct dimensions from the start
# Rows will be cell types, columns will be genes
library_mat <- matrix(0, nrow = length(cell_types_sorted), ncol = nrow(counts_matrix))
rownames(library_mat) <- cell_types_sorted  # Use sorted cell types
colnames(library_mat) <- rownames(counts_matrix)

# Reorder cell_type_mat columns to match sorted order
cell_type_mat <- cell_type_mat[, cell_types_sorted]

# Calculate means using the sorted cell types
for(ct in cell_types_sorted) {  # Use sorted cell types in loop
    cells_in_type <- colnames(seurat_obj)[seurat_obj$leiden == ct]
    library_mat[ct,] <- colMeans(t(counts_matrix[,cells_in_type]))
}

# Ensure library_mat is a matrix
library_mat <- as.matrix(library_mat)

print("Successfully formatted data for NicheDE...")

# Run NicheDE
nichede_obj <- CreateNicheDEObjectFromSeurat(
  seurat_object=seurat_obj,
  assay="RNA",
  sigma=as.integer(snakemake@wildcards[["nichede_niche_size"]]),
  deconv_mat=cell_type_mat,
  library_mat=library_mat
)

# Compute the effective niche
nichede_obj = CalculateEffectiveNiche(nichede_obj)

# Run the niche DE function
nichede_obj = niche_DE_no_parallel(nichede_obj, C=0, M=1, gamma=0)

# Compute the p-values for the niche DE test on positive and negative fold changes
nichede_obj = get_niche_DE_pval_fisher(nichede_obj, pos=T)
nichede_obj = get_niche_DE_pval_fisher(nichede_obj, pos=F)

# Save the p-values for the niche DE test on positive and negative fold changes
positive_pvals <- nichede_obj@niche_DE_pval_pos$interaction_level

# Get dimensions and names
n_cell_types <- ncol(cell_type_mat)
n_genes <- nrow(counts_matrix)
gene_names <- rownames(counts_matrix)
cell_type_names <- colnames(cell_type_mat)

# Function to create interaction dataframe
create_interaction_df <- function(gene_by_cell_type_pvals_pos) {
    gene_by_cell_type_df <- data.frame()

    # Loop through all receivers (i)
    for(i in 1:n_cell_types) {
        # Loop through all senders (j)
        for(j in 1:n_cell_types) {

            # Loop through all genes (k)
            for(k in 1:n_genes) {
                # Create a row with receiver, sender, gene name, and p-values
                row_data <- c(
                    cell_type_names[i],    # receiver
                    cell_type_names[j],    # sender
                    gene_names[k],         # gene name
                    gene_by_cell_type_pvals_pos[i,j,k] # positive p-value
                )
                gene_by_cell_type_df <- rbind(gene_by_cell_type_df, row_data)
            }
        }
    }

    # Add column names
    colnames(gene_by_cell_type_df) <- c("receiver", "sender", "gene", "p_value_adj")
    return(gene_by_cell_type_df)
}

# Create dataframes for positive and negative interactions
gene_by_cell_type_pvals_df <- create_interaction_df(positive_pvals)

# Write to CSV files
write.csv(gene_by_cell_type_pvals_df, snakemake@output[[1]], row.names=FALSE)
