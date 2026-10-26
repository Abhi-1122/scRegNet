import time
import io
import pandas as pd
import requests

STRING_API = "https://version-12-0.string-db.org/api"
SPECIES = 9606
CALLER = "your_project_name"

def map_to_string_ids(genes, species=SPECIES, caller_identity=CALLER, pause=1.0):
    """
    Map gene symbols to STRING IDs.
    Returns:
        mapping_df: full mapping table from STRING
        gene_to_string: dict {preferredName -> stringId}
    """
    url = f"{STRING_API}/tsv/get_string_ids"
    params = {
        "identifiers": "\r".join(genes),
        "species": species,
        "echo_query": 1,
        "caller_identity": caller_identity,
    }
    r = requests.post(url, data=params, timeout=120)
    r.raise_for_status()
    time.sleep(pause)

    mapping_df = pd.read_csv(io.StringIO(r.text), sep="\t")
    # Keep best hit returned by STRING for each preferredName
    mapping_df = mapping_df.drop_duplicates(subset=["preferredName"])
    gene_to_string = dict(zip(mapping_df["preferredName"], mapping_df["stringId"]))
    return mapping_df, gene_to_string


def query_string_network(string_ids, species=SPECIES, required_score=700,
                         network_type="functional", caller_identity=CALLER, pause=1.0):
    """
    Query STRING interactions among the provided STRING IDs.
    required_score is 0-1000 in STRING API.
    network_type: 'functional' or 'physical'
    """
    url = f"{STRING_API}/tsv/network"
    params = {
        "identifiers": "\r".join(string_ids),
        "species": species,
        "required_score": required_score,
        "network_type": network_type,
        "caller_identity": caller_identity,
    }
    r = requests.post(url, data=params, timeout=300)
    r.raise_for_status()
    time.sleep(pause)

    net_df = pd.read_csv(io.StringIO(r.text), sep="\t")
    return net_df


def build_ppi_from_bl_network(
    bl_network_csv,
    species=SPECIES,
    out_prefix="hESC_TFs500_STRING_PPI",
    required_score=700,
    network_type="functional",
):
    # 1) Read existing GRN / node universe
    bl = pd.read_csv(bl_network_csv)
    if not {"Gene1", "Gene2"}.issubset(bl.columns):
        raise ValueError("BL-network.csv must contain columns: Gene1, Gene2")

    genes = sorted(set(bl["Gene1"]).union(set(bl["Gene2"])))
    print(f"Input genes from BL-network: {len(genes)}")

    # 2) Map symbols to STRING IDs
    mapping_df, gene_to_string = map_to_string_ids(genes, species)
    mapped_genes = sorted(gene_to_string.keys())
    mapped_string_ids = sorted(set(gene_to_string.values()))
    print(f"Mapped genes to STRING: {len(mapped_genes)} / {len(genes)}")

    # Save mapping
    mapping_out = f"{out_prefix}.gene_to_string_mapping.csv"
    mapping_df.to_csv(mapping_out, index=False)

    # 3) Query STRING network among mapped genes
    net_df = query_string_network(
        mapped_string_ids,
        species=species,
        required_score=required_score,
        network_type=network_type,
    )
    print(f"Raw STRING interactions returned: {len(net_df)}")

    # 4) Keep only interactions where both preferred names are in our mapped gene universe
    gene_set = set(mapped_genes)
    ppi_df = net_df[
        net_df["preferredName_A"].isin(gene_set) &
        net_df["preferredName_B"].isin(gene_set)
    ].copy()

    # 4b) Report edges under the dataset's own symbol (matters for mouse, where STRING names are Title-case, and for aliases)
    q = dict(zip(mapping_df["preferredName"], mapping_df["queryItem"]))
    ppi_df["preferredName_A"], ppi_df["preferredName_B"] = ppi_df["preferredName_A"].map(q), ppi_df["preferredName_B"].map(q)

    # 5) Remove self-loops
    ppi_df = ppi_df[ppi_df["preferredName_A"] != ppi_df["preferredName_B"]].copy()

    # 6) Deduplicate as undirected edges
    ppi_df["node_u"] = ppi_df[["preferredName_A", "preferredName_B"]].min(axis=1)
    ppi_df["node_v"] = ppi_df[["preferredName_A", "preferredName_B"]].max(axis=1)
    ppi_df = ppi_df.sort_values("score", ascending=False).drop_duplicates(
        subset=["node_u", "node_v"]
    )

    # 7) Save full weighted edge list
    weighted_cols = [
        "node_u", "node_v", "score",
        "nscore", "fscore", "pscore", "ascore",
        "escore", "dscore", "tscore",
        "stringId_A", "stringId_B"
    ]
    weighted_out = f"{out_prefix}.weighted_edges.csv"
    ppi_df[weighted_cols].to_csv(weighted_out, index=False)

    # 8) Save simple edge list
    simple_out = f"{out_prefix}.edges.csv"
    ppi_df[["node_u", "node_v"]].to_csv(simple_out, index=False, header=["Gene1", "Gene2"])

    # 9) Optional: make symmetric edge list for PyG / directed message passing
    sym_df = pd.concat([
        ppi_df[["node_u", "node_v", "score"]].rename(columns={"node_u": "Gene1", "node_v": "Gene2"}),
        ppi_df[["node_v", "node_u", "score"]].rename(columns={"node_v": "Gene1", "node_u": "Gene2"})
    ], ignore_index=True)
    sym_out = f"{out_prefix}.symmetric_edges.csv"
    sym_df.to_csv(sym_out, index=False)

    print("\nDone.")
    print(f"Mapping file:          {mapping_out}")
    print(f"Weighted edge list:    {weighted_out}")
    print(f"Simple edge list:      {simple_out}")
    print(f"Symmetric edge list:   {sym_out}")
    print(f"Final undirected PPI edges: {len(ppi_df)}")

    return {
        "mapping": mapping_out,
        "weighted_edges": weighted_out,
        "simple_edges": simple_out,
        "symmetric_edges": sym_out,
        "num_input_genes": len(genes),
        "num_mapped_genes": len(mapped_genes),
        "num_ppi_edges": len(ppi_df),
    }


if __name__ == "__main__":
    # usage: python3 make_ppi.py [CELL SPECIES]   e.g. mESC 10090 (default: hESC 9606)
    import sys
    cell, species = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) > 2 else ("hESC", SPECIES)
    result = build_ppi_from_bl_network(
        bl_network_csv=f"./data/{cell}/TFs+500/BL--network.csv",
        species=species,
        out_prefix=f"./data/{cell}/TFs+500/STRING/{cell}_TFs500_STRING_PPI",
        required_score=700,          # change this
        network_type="functional",   # or "physical"
    )
    print(result)
