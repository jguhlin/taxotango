use burn::data::dataset::{Dataset, DatasetIterator};
use crossbeam::channel::bounded;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_xoshiro::Xoshiro256PlusPlus;
// use rerun::Color;

use std::fs::File;
use std::hash::Hash;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Arc;
use std::thread::JoinHandle;

use fnv::FnvHashMap as HashMap;
use fnv::FnvHashSet as HashSet;

const MAMMALIA_DEBUG: bool = false;
const MAMMAL_NOUN_EXAMPLE: bool = false;

pub mod model;
pub use model::*;

pub mod layers;
pub use layers::*;

pub mod optimizers;
pub use optimizers::*;

pub enum TaxonomyWriterMessage {
    Write(Vec<(NodeIndex, NodeIndex, u8)>),
    Completed,
}

#[derive(PartialEq, PartialOrd, Eq, Ord, Debug, Clone, Copy, Hash)]
pub enum TaxaLevel {
    Root,
    Superkingdom,
    Kingdom,
    Subkingdom,
    Superphylum,
    Phylum,
    Subphylum,
    Superclass,
    Class,
    Subclass,
    Infraclass,
    Cohort,
    Subcohort,
    Superorder,
    Order,
    Suborder,
    Infraorder,
    Parvorder,
    Superfamily,
    Family,
    Subfamily,
    Tribe,
    Subtribe,
    Genus,
    Subgenus,
    SpeciesGroup,
    SpeciesSubgroup,
    Species,
    Subspecies,
    Varietas,
    Forma,
    Section,
    Subsection,
    Series,
    Clade,
    NoRank,
}

impl TaxaLevel {
    pub fn from_str(rank: &str) -> Self {
        match rank {
            "no rank" => Self::NoRank,
            "species" => Self::Species,
            "tribe" => Self::Tribe,
            "genus" => Self::Genus,
            "superfamily" => Self::Superfamily,
            "family" => Self::Family,
            "subfamily" => Self::Subfamily,
            "order" => Self::Order,
            "infraorder" => Self::Infraorder,
            "suborder" => Self::Suborder,
            "class" => Self::Class,
            "subclass" => Self::Subclass,
            "phylum" => Self::Phylum,
            "subphylum" => Self::Subphylum,
            "kingdom" => Self::Kingdom,
            "superkingdom" => Self::Superkingdom,
            "root" => Self::Root,
            "subspecies" => Self::Subspecies,
            "species group" => Self::SpeciesGroup,
            "subgenus" => Self::Subgenus,
            "clade" => Self::Clade,
            "forma" => Self::Forma,
            "varietas" => Self::Varietas,
            "infraclass" => Self::Infraclass,
            "superorder" => Self::Superorder,
            "superclass" => Self::Superclass,
            "parvorder" => Self::Parvorder,
            "species subgroup" => Self::SpeciesSubgroup,
            "subcohort" => Self::Subcohort,
            "cohort" => Self::Cohort,
            "subtribe" => Self::Subtribe,
            "section" => Self::Section,
            "series" => Self::Series,
            "subkingdom" => Self::Subkingdom,
            "superphylum" => Self::Superphylum,
            "subsection" => Self::Subsection,
            _ => panic!("Unknown rank: {}", rank),
        }
    }

    /*
    pub fn color(&self) -> Color {
        match self {
            TaxaLevel::NoRank => Color::from_rgb(128, 128, 128), // Gray
            // TaxaLevel::Root => Color::from_rgb(0, 0, 0),         // Black
            TaxaLevel::Root => Color::from_rgb(255, 255, 255), // White
            TaxaLevel::Superkingdom => Color::from_rgb(255, 105, 180), // Hot Pink
            TaxaLevel::Kingdom => Color::from_rgb(255, 20, 147), // Deep Pink
            TaxaLevel::Subkingdom => Color::from_rgb(200, 20, 110), // Darker Deep Pink
            TaxaLevel::Superphylum => Color::from_rgb(255, 150, 255), // Lighter Violet
            TaxaLevel::Phylum => Color::from_rgb(238, 130, 238), // Violet
            TaxaLevel::Subphylum => Color::from_rgb(200, 100, 200), // Darker Violet
            TaxaLevel::Superclass => Color::from_rgb(100, 0, 160), // Lighter Indigo
            // TaxaLevel::Class => Color::from_rgb(75, 0, 130),     // Indigo
            TaxaLevel::Class => Color::from_rgb(255, 255, 255), // White
            TaxaLevel::Infraclass => Color::from_rgb(60, 0, 110), // Darker Indigo
            TaxaLevel::Subclass => Color::from_rgb(60, 0, 110), // Darker Indigo
            TaxaLevel::Superorder => Color::from_rgb(255, 255, 100), // Lighter Yellow
            TaxaLevel::Order => Color::from_rgb(255, 255, 0),   // Yellow
            TaxaLevel::Parvorder => Color::from_rgb(200, 200, 0), // Darker Yellow
            TaxaLevel::Infraorder => Color::from_rgb(200, 200, 0), // Darker Yellow
            TaxaLevel::Suborder => Color::from_rgb(200, 200, 0), // Darker Yellow
            TaxaLevel::Superfamily => Color::from_rgb(255, 215, 0), // Gold
            TaxaLevel::Family => Color::from_rgb(255, 165, 0),  // Orange
            TaxaLevel::Subfamily => Color::from_rgb(255, 140, 0), // Darker Orange
            TaxaLevel::Tribe => Color::from_rgb(0, 0, 255),     // Bright Blue
            TaxaLevel::Subtribe => Color::from_rgb(0, 0, 200),  // Darker Blue
            TaxaLevel::Genus => Color::from_rgb(0, 128, 0),     // Green
            TaxaLevel::Subgenus => Color::from_rgb(0, 100, 0),  // Darker Green
            TaxaLevel::SpeciesGroup => Color::from_rgb(200, 0, 0), // Darker Red
            TaxaLevel::Species => Color::from_rgb(255, 0, 0),   // Bright Red
            TaxaLevel::Subspecies => Color::from_rgb(200, 0, 0), // Darker Red
            TaxaLevel::Clade => Color::from_rgb(128, 0, 128),   // Purple
            TaxaLevel::Forma => Color::from_rgb(255, 20, 147),  // Deep Pink
            TaxaLevel::Varietas => Color::from_rgb(255, 20, 147), // Deep Pink
            TaxaLevel::SpeciesSubgroup => Color::from_rgb(200, 0, 0), // Darker Red
            TaxaLevel::Subcohort => Color::from_rgb(200, 50, 0), // Darker Red-Orange
            TaxaLevel::Cohort => Color::from_rgb(255, 69, 0),   // Red-Orange
            TaxaLevel::Section => Color::from_rgb(34, 139, 34), // Forest Green
            TaxaLevel::Subsection => Color::from_rgb(34, 139, 34), // Forest Green
            TaxaLevel::Series => Color::from_rgb(34, 139, 34),  // Forest Green
        }
    }
     */
}

pub type TaxonomyGraph = Graph<Taxon, (), Directed, u32>;

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct Taxon {
    pub tax_id: u32,
    pub parent: u32,
    pub rank: TaxaLevel,
    pub name: String,
    // pub color: Color,
    pub rank_str: String,
}

impl Hash for Taxon {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tax_id.hash(state);
    }
}

pub fn build_taxonomy_graph(nodes_file: &str, names_file: &str) -> (TaxonomyGraph, NodeIndex) {
    let taxa_names = parse_names(names_file.to_string());
    let nodes = parse_nodes(nodes_file.to_string());

    let taxa_parents: HashMap<u32, u32> = nodes.0.iter().map(|(x, y)| (*x, *y)).collect();
    let taxa_ranks: HashMap<u32, String> = nodes
        .1
        .iter()
        .zip(nodes.0.iter())
        .map(|(x, y)| (y.0, x.clone()))
        .collect();

    assert!(
        taxa_names.get(&271808).unwrap() == "Wajira",
        "Taxa name is not correct: {}",
        taxa_names.get(&271808).unwrap()
    );

    let edges: Vec<(u32, u32)> = nodes.0;

    let nodes: Vec<u32> = edges.iter().flat_map(|(x, y)| [x, y]).copied().collect();
    let nodes = nodes.into_iter().collect::<HashSet<u32>>();
    let mut nodes = nodes.into_iter().collect::<Vec<u32>>();
    nodes.sort();

    assert!(nodes[0] == 1, "Root node is not 1");

    assert!(
        nodes.len() == taxa_names.len(),
        "Nodes and names are not the same length"
    );

    // let mut graph = DiGraph::<u32, (), u32>::with_capacity(nodes.len(), nodes.len());
    let mut graph = DiGraph::<Taxon, (), u32>::with_capacity(nodes.len(), nodes.len());

    let nodes = nodes
        .into_iter()
        .map(|x| Taxon {
            tax_id: x,
            parent: taxa_parents[&x],
            rank: TaxaLevel::from_str(&taxa_ranks[&x]),
            name: taxa_names[&x].clone(),
            // color: TaxaLevel::from_str(&taxa_ranks[&x]).color(),
            rank_str: taxa_ranks[&x].clone(),
        })
        // If tax_id == 1 set rank to Root
        .map(|x| {
            if x.tax_id == 1 {
                Taxon {
                    rank: TaxaLevel::Root,
                    ..x
                }
            } else {
                x
            }
        })
        // .filter(|x| x.rank != TaxaLevel::NoRank)
        .collect::<Vec<_>>();

    let nodes_set = nodes.iter().map(|x| x.tax_id).collect::<HashSet<_>>();

    assert!(nodes_set.len() > 0, "Filtered out all nodes");
    assert!(nodes_set.contains(&1), "Root node not in set");

    // Update edges to only those that are in the set
    let edges = edges
        .into_iter()
        .filter(|(x, y)| nodes_set.contains(x) && nodes_set.contains(y))
        .collect::<Vec<_>>();

    let edges_set = edges
        .iter()
        .flat_map(|(x, y)| [*x, *y])
        .collect::<HashSet<_>>();

    // Make sure all nodes have an edge
    let nodes = nodes
        .into_iter()
        .filter(|x| edges_set.contains(&x.tax_id))
        .collect::<Vec<_>>();

    println!("Length of nodes: {}", nodes.len());

    for node in nodes.iter() {
        graph.add_node(node.clone());
    }

    let node_indices = graph
        .node_indices()
        .map(|x| (graph[x].tax_id, x))
        .collect::<HashMap<_, _>>();

    for (x, y) in edges {
        // x and y must still be in the set
        if !node_indices.contains_key(&x) || !node_indices.contains_key(&y) {
            continue;
        }

        // No circular edges
        if x == y {
            continue;
        }

        graph.add_edge(node_indices[&y], node_indices[&x], ());
    }

    let root = node_indices[&1];

    // Just for now, make a new graph of only Mammalia (40674) and its children, and train on that

    let (graph, root) = if MAMMALIA_DEBUG {
    
        let top_id = 40674;
        let top_index = node_indices[&top_id];

        let mut new_graph = DiGraph::<Taxon, (), u32>::with_capacity(nodes.len(), nodes.len());

        let mut new_nodes = HashSet::default();
        let mut new_edges = Vec::new();

        let mut stack = vec![top_index];
        new_nodes.insert(top_id);

        while let Some(node) = stack.pop() {
            let children = graph.neighbors_directed(node, Direction::Outgoing);
            for child in children {
                let child_id = graph[child].tax_id;
                new_nodes.insert(child_id);
                new_edges.push((graph[node].tax_id, child_id));
                stack.push(child);
            }
        }

        for node in new_nodes.iter() {
            let node = node_indices[node];
            new_graph.add_node(graph[node].clone());
        }

        let new_node_indices = new_graph
            .node_indices()
            .map(|x| (new_graph[x].tax_id, x))
            .collect::<HashMap<_, _>>();

        for (x, y) in new_edges {
            new_graph.add_edge(new_node_indices[&x], new_node_indices[&y], ());
        }

        let node_indices = new_graph
            .node_indices()
            .map(|x| (new_graph[x].tax_id, x))
            .collect::<HashMap<_, _>>();

        graph = new_graph;

        let root = node_indices[&top_id];

        log::info!(
            "Taxonomy graph built. Node Count: {} - Edge Count: {}",
            graph.node_count(),
            graph.edge_count()
        );

        (graph, root)
    } else {
        (graph, root)
    };

    println!("Length of nodes: {}", graph.node_count());
    
    // Find "Rattini" (27564)
    println!("Rattini Index: {:?}", node_indices[&39107]);

    println!("Root Graph Index is {}", root.index());

    (graph, root)
}

pub fn build_taxonomy_graph_generator<const P: usize, const N: usize>(
    nodes_file: &str,
    names_file: &str,
    threads: usize,
) -> BatchGenerator<P, N> {
    let (graph, root) = build_taxonomy_graph(nodes_file, names_file);
    let (tx, rx) = bounded(8192 * 128);

    let graph = Arc::new(graph);

    let node_indices = Arc::new(
        graph
            .node_indices()
            .map(|x| (graph[x].tax_id, x))
            .collect::<HashMap<_, _>>(),
    );

    // Let's count each taxon so we can weight them
    let mut taxon_counts = HashMap::default();

    let mut total = 0;
    for edge_index in graph.edge_indices() {
        let edge = graph.edge_endpoints(edge_index).unwrap();
        let parent = graph[edge.0].tax_id;
        let child = graph[edge.1].tax_id;

        *taxon_counts.entry(parent).or_insert(0) += 1;
        *taxon_counts.entry(child).or_insert(0) += 1;
        total += 2;
    }

    // Get the max taxon count
    let max_taxon_count = *taxon_counts.values().max().unwrap();

    // Set the root to the max count + 10 so it's picked frequently
    if MAMMALIA_DEBUG {
        *taxon_counts.get_mut(&40674).unwrap() = 0;
    } else {
        *taxon_counts.get_mut(&1).unwrap() = max_taxon_count;
    }

    // Calc distance to root, then weight by inverse distance
    let mut taxon_dist_to_root = taxon_counts
        .iter()
        .map(|(tax_id, count)| {
            let dist = calc_dist_to_root(Arc::as_ref(&graph), node_indices[tax_id], root);
            (*tax_id, dist as u32)
        })
        .collect::<HashMap<_, _>>();

    let max_dist = *taxon_dist_to_root.values().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    // Let's calculate dist to root per taxon level (rank) except for NoRank, which we can leave out
    // Let's do this by getting the max dist for each rank
    let mut rank_dist_to_root = HashMap::default();

    for (tax_id, dist) in taxon_dist_to_root.iter() {
        let rank = graph[node_indices[tax_id]].rank;
        let max_dist = rank_dist_to_root.entry(rank).or_insert(0);
        *max_dist = std::cmp::max(*max_dist, *dist);
    }

    // Remove NoRank
    rank_dist_to_root.remove(&TaxaLevel::NoRank);

    // Update taxon_dist_to_root to be the max dist for the rank
    for (tax_id, dist) in taxon_dist_to_root.iter_mut() {
        let rank = graph[node_indices[tax_id]].rank;
        if rank == TaxaLevel::NoRank {
            continue;
        }
        *dist = *rank_dist_to_root.get(&rank).unwrap();
    }

    let sample_weights = taxon_counts.iter()
    .map(|(tax_id, count)| {
        let dist = taxon_dist_to_root[tax_id];
        let weight = 1.0 - (1.0 / (1.0 + dist as f64));
        (*tax_id, weight)
    }).collect::<HashMap<_, _>>();

    // Normalize samples weight to 0 to 1
    let sample_weights_max = *sample_weights.values().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let sample_weights = sample_weights.iter().map(|(tax_id, weight)| {
        (*tax_id, weight / sample_weights_max)
    }).collect::<HashMap<_, _>>();

    // Use sample_weights as the distant weight
    let distant_weights = sample_weights.clone();

    let distant_weights_tax_ids = Arc::new(distant_weights
        .iter()
        .map(|(tax_id, _)| *tax_id)
        .collect::<Vec<_>>());

    let distant_weights = Arc::new(WeightedIndex::new(distant_weights.iter().map(|(_, weight)| *weight)).unwrap());

    // Formulae (max dist - dists) / max dist

    // Normalize to 0-1 (with 1 being the root)
    let taxon_dist_to_root_weights = taxon_counts
        .iter()
        .map(|(tax_id, count)| {
            let dist = calc_dist_to_root(Arc::as_ref(&graph), node_indices[tax_id], root);
            (*tax_id, 1.0 / (1.0 + dist as f64))
        })
        .collect::<HashMap<_, _>>();

    // Let's try weights by count (connectedness)
    let mut taxon_weights = taxon_counts
        .iter()
        .map(|(tax_id, count)| {
            (*tax_id, *count as f64 / total as f64)
        })
        .collect::<HashMap<_, _>>();

    // for (tax_id, dist) in taxon_dist_to_root_weights.iter() {
        // let count = taxon_weights[tax_id];
        // taxon_weights.insert(*tax_id, count + dist);
        // taxon_weights.insert(*tax_id, *dist);
    // }

    // Set root to 0
    /*
    if MAMMALIA_DEBUG {
        let weight = taxon_weights.get_mut(&40674).unwrap();
        *weight = 0.0;
    } else {
        let weight = taxon_weights.get_mut(&1).unwrap();
        *weight = 0.0;
    }  */

    // Using node_indices, conver sample_weights to node_indices and a vec
    let node_indices_max = node_indices.len();
    let mut sample_weights_final = vec![0.0; node_indices_max];
    for (tax_id, weight) in sample_weights.iter() {
        let idx = node_indices[tax_id];
        sample_weights_final[idx.index()] = *weight;
    }

    let sample_weights = sample_weights_final.iter().map(|x| *x as f32).collect::<Vec<_>>();
        
    // Root has a normalization factor now, so we don't have to adjust it manually

    // Remove those with no children
    taxon_weights.retain(|tax_id, _| {
        let idx = node_indices[tax_id];
        graph.neighbors_directed(idx, Direction::Outgoing).count() > 0
    });

    let taxon_weights_raw = taxon_weights.into_iter().collect::<Vec<_>>();
    println!("Taxon Weights Count: {}", taxon_weights_raw.len());

    // Min and max
    println!(
        "Min: {:?} - Max: {:?}",
        taxon_weights_raw.iter().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()),
        taxon_weights_raw.iter().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    );
    let taxon_weights =
        Arc::new(WeightedIndex::new(taxon_weights_raw.iter().map(|(_, weight)| *weight)).unwrap());

    let weighted_tax_ids = taxon_weights_raw
        .iter()
        .map(|(tax_id, _)| *tax_id)
        .collect::<Vec<_>>();

    // Did just the weighted ones before (only those with outgoing edges) but it didn't play nice
    let tax_ids = Arc::new(
        taxon_counts
            .into_iter()
            .map(|(tax_id, _)| tax_id)
            .collect::<Vec<_>>(),
    );

    // Spawn threads
    let mut jhs = Vec::with_capacity(threads);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337);
    let shutdown = Arc::new(AtomicBool::new(false));
    let weighted_sampling = Arc::new(AtomicBool::new(true));
    let current = Arc::new(AtomicUsize::new(0));
    let weighted_tax_ids = Arc::new(weighted_tax_ids);
    let taxon_dist_to_root = Arc::new(taxon_dist_to_root);

    log::info!("Weighted Tax IDs Count: {}", weighted_tax_ids.len());
    log::info!("Tax IDs Count: {}", tax_ids.len());

    let all_nodes = Arc::new(graph.node_indices().collect::<Vec<_>>());

    // Print first 10
    log::debug!("First 10 Nodes: {:?}", &all_nodes[0..10]);

    // Print last 10
    log::debug!("Last 10 Nodes: {:?}", &all_nodes[all_nodes.len() - 10..]);

    println!("Max Dist: {}", max_dist);

    for _threadno in 0..threads {
        let tx = tx.clone();
        let graph = Arc::clone(&graph);
        let shutdown = Arc::clone(&shutdown);
        rng.long_jump();
        let mut rng = rng.clone();
        let current = Arc::clone(&current);
        let all_nodes = Arc::clone(&all_nodes);
        let weighted_sampling = Arc::clone(&weighted_sampling);
        let taxon_weights = Arc::clone(&taxon_weights);
        let tax_ids = Arc::clone(&tax_ids);
        let weighted_tax_ids = Arc::clone(&weighted_tax_ids);
        let node_indices = Arc::clone(&node_indices);
        let taxon_dist_to_root = Arc::clone(&taxon_dist_to_root);
        let max_dist = max_dist;
        let distant_weights = Arc::clone(&distant_weights);
        let distant_weights_tax_ids = Arc::clone(&distant_weights_tax_ids);

        let jh = std::thread::spawn(move || {
            // let mut local_excluded = HashSet::default();
            let mut nearby = [0; P];
            let mut distant = [0; N];
            let mut origin_weight_factor;
            let mut nearby_weight_factor = [0.0; P];
            let mut distant_weight_factor = [0.0; N];

            loop {
                if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }

                let idx = if weighted_sampling.load(std::sync::atomic::Ordering::Relaxed) {
                    weighted_tax_ids[taxon_weights.sample(&mut rng)]
                } else {
                    *tax_ids.choose(&mut rng).unwrap()
                };

                let tax_id = idx;

                let mut idx = node_indices[&idx];

                origin_weight_factor = (1.0 + max_dist as f32 - taxon_dist_to_root[&tax_id] as f32) / max_dist as f32;
                origin_weight_factor = 1.0 - origin_weight_factor;

                let outgoing_neighbors = graph.neighbors_directed(idx, Direction::Outgoing);

                // If idx is an end node (has no children), change to the parent
                if outgoing_neighbors.count() == 0 {
                    // Switch to parent
                    let parent = graph
                        .neighbors_directed(idx, Direction::Incoming)
                        .next()
                        .unwrap();
                    idx = parent;
                }

                let outgoing_neighbors = graph.neighbors_directed(idx, Direction::Outgoing);
                let outgoing_neighbors = outgoing_neighbors.collect::<Vec<_>>();

                let neighbors = graph.neighbors_undirected(idx);
                let mut neighbors = neighbors.collect::<Vec<_>>();

                let mut query_neighbors = neighbors.clone();
//                let mut additional_neighbors = Vec::new();
                // Get neighbor's neighbors
/*                for _ in 0..1 { // 3 for mammalia, but too slow if it's 3 for all...
                    for neighbor in query_neighbors.iter() {
                        let neighbor_neighbors = graph.neighbors_undirected(*neighbor);
                        additional_neighbors.extend(neighbor_neighbors);
                    }
                    neighbors.extend_from_slice(&additional_neighbors);
                    std::mem::swap(&mut query_neighbors, &mut additional_neighbors);
                    additional_neighbors.clear();
                }

                */

                // If we need more, get more...
                /* while neighbors.len() < P && query_neighbors.len() < P {
                    for neighbor in query_neighbors.iter() {
                        let neighbor_neighbors = graph.neighbors_undirected(*neighbor);
                        additional_neighbors.extend(neighbor_neighbors);
                    }
                    neighbors.extend_from_slice(&additional_neighbors);
                    std::mem::swap(&mut query_neighbors, &mut additional_neighbors);
                    additional_neighbors.clear();
                } */

                assert!(neighbors.len() > 0, "Node has no neighbors");
                // Pick a random neighbor
                // Only children
                let node = outgoing_neighbors.choose(&mut rng).clone();
                let node_tax_id = graph.raw_nodes()[node.unwrap().index()].weight.tax_id;
                nearby[0] = node.unwrap().index() as u32;
                nearby_weight_factor[0] = (1.0 + max_dist as f32 - taxon_dist_to_root[&node_tax_id] as f32) / max_dist as f32;
                nearby_weight_factor[0] = 1.0 - nearby_weight_factor[0];

                // Pick P-1 random nearby neighbors
                for i in 1..P {
                    let choose_from = if outgoing_neighbors.len() <= P {
                        &outgoing_neighbors
                    } else if query_neighbors.len() <= P {
                        &query_neighbors
                    } else {
                        &neighbors
                    };
                    let node = choose_from.choose(&mut rng).unwrap();

                    let node_tax_id = graph.raw_nodes()[node.index()].weight.tax_id;
                    nearby_weight_factor[i] = (1.0 + max_dist as f32 - taxon_dist_to_root[&node_tax_id] as f32) / max_dist as f32;
                    nearby_weight_factor[i] = 1.0 - nearby_weight_factor[i];
                    nearby[i] = node.index() as u32;
                }

                if P > 1 {
                    // Add the parent as a nearby (if it exists)
                    let parent = graph
                        .neighbors_directed(idx, Direction::Incoming)
                        .next();

                    if let Some(parent) = parent {
                        nearby[1] = parent.index() as u32;
                    } // no else, this slot is chosen by the above loop
                }

                /*

                if P > 2 {
                    // Pick the grandparent as a nearby (if it exists)
                    let grandparent = graph
                        .neighbors_directed(idx, Direction::Incoming)
                        .next()
                        .and_then(|x| graph.neighbors_directed(x, Direction::Incoming).next());

                    if let Some(grandparent) = grandparent {
                        nearby[2] = grandparent.index() as u32;
                    } // no else, this slot is chosen by the above loop
                }  */

                // todo calc path to root for origin and nearbyy
                // and find if it intersects with distant ones at the root, and remove it
                // Although the chance of it happening is very low....
                // Or maybe it's a BFS.. although that is technically what we are doing with the for
                // loop above
                                
                for i in 0..N {
                    let mut end = distant_weights_tax_ids[distant_weights.sample(&mut rng)];
                    let mut end = node_indices[&end];

                    // Make sure it isn't a neighbor
                    while neighbors.contains(&end)
                        || end == idx
                        || end == root
                        // if MAMMALIA_DEBUG then it's the full dataset, so ignore...
                        || (MAMMALIA_DEBUG && is_descendant(Arc::as_ref(&graph), idx, end, root))
                        || (MAMMALIA_DEBUG && is_descendant(Arc::as_ref(&graph), end, idx, root))
                    {
                        end = node_indices[&distant_weights_tax_ids[distant_weights.sample(&mut rng)]];
                    }

                    // let end_dist_to_root = dist_to_root(Arc::as_ref(&graph), end, root);
                    distant[i] = end.index() as u32;

                    let end_tax_id = graph.raw_nodes()[end.index()].weight.tax_id;
                    distant_weight_factor[i] = (1.0 + max_dist as f32 - taxon_dist_to_root[&end_tax_id] as f32) / max_dist as f32;
                    distant_weight_factor[i] = 1.0 - distant_weight_factor[i];
                }

                // println!("Nearby Weight Factors: {:?}", nearby_weight_factor);
                // println!("Distant Weight Factors: {:?}", distant_weight_factor);
                // println!("Origin Weight Factor: {:?}", origin_weight_factor);

                let taxa_dist = TaxaDistance {
                    origin: idx.index() as u32,
                    nearby,
                    distant,
                    origin_weight_factor,
                    nearby_weight_factor,
                    distant_weight_factor,

                };

                match tx.send(taxa_dist) {
                    Ok(_) => {}
                    Err(_) => {
                        println!("Error sending taxa distance");
                        return;
                    }
                }
            }
        });

        jhs.push(jh);
    }

    BatchGenerator {
        root,
        epoch_size: graph.node_count(),
        graph,
        join_handles: jhs,
        shutdown,
        rx,
        weighted_sampling,
        sample_weights,
        // taxa_dist_cache,
    }
}

pub struct BatchGenerator<const P: usize, const N: usize> {
    pub root: NodeIndex,
    pub graph: Arc<Graph<Taxon, (), Directed, u32>>,
    epoch_size: usize,
    join_handles: Vec<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    rx: crossbeam::channel::Receiver<TaxaDistance<P, N>>,
    pub weighted_sampling: Arc<AtomicBool>,
    pub sample_weights: Vec<f32>,
    // pub taxa_dist_cache: Arc<TaxaDistCache<D>>,
    
}

impl<const P: usize, const N: usize> Dataset<TaxaDistance<P, N>> for BatchGenerator<P, N> {
    fn len(&self) -> usize {
        self.epoch_size
    }

    fn get(&self, _index: usize) -> Option<TaxaDistance<P, N>> {
        // self.receiver.recv().ok()
        // self.taxa_dist_cache.get(index).into()
        let rx = self.rx.recv().ok();

        return rx
    }

    // Provided methods
    fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    fn iter(&self) -> DatasetIterator<'_, TaxaDistance<P, N>>
    where
        Self: Sized,
    {
        DatasetIterator::new(self)
    }
}

impl<const P: usize, const N: usize> BatchGenerator<P, N> {
    pub fn precache(&self) {
        // Optimal is 10%
        let cache_warmup_size = self.graph.node_count() / 5;

        // In development mode is less though...
        // let cache_warmup_size = 2048;

        /*
        while self.taxa_dist_cache.len() < cache_warmup_size {
            println!(
                "Waiting for cache to fill: {}/{}",
                self.taxa_dist_cache.len(),
                cache_warmup_size
            );
            std::thread::sleep(std::time::Duration::from_secs(1));
        } */

        while self.rx.len() < cache_warmup_size {
            println!(
                "Waiting for cache to fill: {}/{}",
                self.rx.len(),
                cache_warmup_size
            );
            std::thread::sleep(std::time::Duration::from_secs(1));
        }

        println!("Cache filled");
    }

    pub fn shutdown(&mut self) -> Result<(), &'static str> {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);

        let mut jhs = Vec::new();
        std::mem::swap(&mut jhs, &mut self.join_handles);

        for jh in jhs {
            match jh.join() {
                Ok(_) => {}
                Err(_) => {
                    return Err("Error joining batch generator thread");
                }
            }
        }

        Ok(())
    }
    
    pub fn valid(&self) -> Self {
        BatchGenerator {
            root: self.root,
            graph: Arc::clone(&self.graph),
            epoch_size: 2048,
            join_handles: vec![],
            shutdown: Arc::clone(&self.shutdown),
            rx: self.rx.clone(),
            weighted_sampling: Arc::clone(&self.weighted_sampling),
            sample_weights: self.sample_weights.clone(),
            // taxa_dist_cache: Arc::clone(&self.taxa_dist_cache),
        }
    }

    pub fn taxonomy_size(&self) -> usize {
        self.graph.node_count()
    }
}

fn is_descendant(
    graph: &TaxonomyGraph,
    origin_node: NodeIndex,
    query_node: NodeIndex,
    root: NodeIndex,
) -> bool {
    // Go up the tree from the query node to the root, if we hit the origin node, it's a descendant
    // and we return true

    let mut current = query_node;

    while current != root {
        if current == origin_node {
            return true;
        }

        let parent = graph
            .neighbors_directed(current, Direction::Incoming)
            .next()
            .unwrap();
        current = parent;
    }

    false
}

// Type alias
pub type TaxId = u32;
pub type TaxonRank = String;

pub fn parse_nodes(filename: String) -> (Vec<(TaxId, TaxId)>, Vec<TaxonRank>) {
    let mut taxon_to_parent: Vec<(u32, u32)> = Vec::with_capacity(4_000_000);
    let mut taxon_rank: Vec<String> = Vec::with_capacity(4_000_000);

    let reader = BufReader::new(File::open(filename).expect("Unable to open taxonomy names file"));

    let lines = reader.lines();

    for line in lines {
        let split = line
            .expect("Error reading line")
            .split('|')
            .map(|x| x.trim().to_string())
            .collect::<Vec<String>>();

        let tax_id: u32 = split[0].parse().expect("Error converting to number");
        let parent_id: u32 = split[1].parse().expect("Error converting to number");
        let rank: &str = &split[2];

        taxon_to_parent.push((tax_id, parent_id));
        taxon_rank.push(rank.into());
    }

    taxon_to_parent.shrink_to_fit();
    taxon_rank.shrink_to_fit();

    (taxon_to_parent, taxon_rank)
}

pub type TaxonName = String;

pub fn parse_names(filename: String) -> HashMap<TaxId, TaxonName> {
    let mut names = HashMap::default();

    let reader = BufReader::new(File::open(filename).expect("Unable to open taxonomy names file"));

    let lines = reader.lines();
    let mut taxids = HashSet::default();

    for line in lines {
        let split = line
            .expect("Error reading line")
            .split('|')
            .map(|x| x.trim().to_string())
            .collect::<Vec<String>>();

        let id: usize = split[0].parse().expect("Error converting to number");
        let name: &str = &split[1];
        // let class: &str = &split[3];

        // Set if only not set (some have multiple names)
        if !taxids.contains(&id) {
            names.insert(id as u32, name.to_string());
            taxids.insert(id);
        }
    }

    names
}

fn calc_dist_to_root(graph: &TaxonomyGraph, node: NodeIndex, root: NodeIndex) -> usize {
    let mut current = node;
    let mut dist = 0;

    while current != root {
        let parent = graph
            .neighbors_directed(current, Direction::Incoming)
            .next()
            .unwrap();
        current = parent;
        dist += 1;
    }

    dist
}
