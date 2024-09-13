use bumpalo::Bump;
use burn::data::dataset::{Dataset, DatasetIterator};
use crossbeam::channel::{bounded, unbounded};
use petgraph::adj::NodeIndices;
use petgraph::algo::astar;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::prelude::*;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_xoshiro::Xoshiro256PlusPlus;
use rerun::Color;

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;

use fnv::FnvHashMap as HashMap;
use fnv::FnvHashSet as HashSet;

pub mod model;
pub use model::*;

pub mod layers;
pub use layers::*;

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

    pub fn color(&self) -> Color {
        match self {
            TaxaLevel::NoRank => Color::from_rgb(128, 128, 128), // Gray
            TaxaLevel::Root => Color::from_rgb(0, 0, 0),         // Black
            TaxaLevel::Superkingdom => Color::from_rgb(255, 105, 180), // Hot Pink
            TaxaLevel::Kingdom => Color::from_rgb(255, 20, 147), // Deep Pink
            TaxaLevel::Subkingdom => Color::from_rgb(200, 20, 110), // Darker Deep Pink
            TaxaLevel::Superphylum => Color::from_rgb(255, 150, 255), // Lighter Violet
            TaxaLevel::Phylum => Color::from_rgb(238, 130, 238), // Violet
            TaxaLevel::Subphylum => Color::from_rgb(200, 100, 200), // Darker Violet
            TaxaLevel::Superclass => Color::from_rgb(100, 0, 160), // Lighter Indigo
            TaxaLevel::Class => Color::from_rgb(75, 0, 130),     // Indigo
            TaxaLevel::Infraclass => Color::from_rgb(60, 0, 110), // Darker Indigo
            TaxaLevel::Subclass => Color::from_rgb(60, 0, 110),  // Darker Indigo
            TaxaLevel::Superorder => Color::from_rgb(255, 255, 100), // Lighter Yellow
            TaxaLevel::Order => Color::from_rgb(255, 255, 0),    // Yellow
            TaxaLevel::Parvorder => Color::from_rgb(200, 200, 0), // Darker Yellow
            TaxaLevel::Infraorder => Color::from_rgb(200, 200, 0), // Darker Yellow
            TaxaLevel::Suborder => Color::from_rgb(200, 200, 0), // Darker Yellow
            TaxaLevel::Superfamily => Color::from_rgb(255, 215, 0), // Gold
            TaxaLevel::Family => Color::from_rgb(255, 165, 0),   // Orange
            TaxaLevel::Subfamily => Color::from_rgb(255, 140, 0), // Darker Orange
            TaxaLevel::Tribe => Color::from_rgb(0, 0, 255),      // Blue
            TaxaLevel::Subtribe => Color::from_rgb(0, 0, 200),   // Darker Blue
            TaxaLevel::Genus => Color::from_rgb(0, 128, 0),      // Green
            TaxaLevel::Subgenus => Color::from_rgb(0, 100, 0),   // Darker Green
            TaxaLevel::SpeciesGroup => Color::from_rgb(200, 0, 0), // Darker Red
            TaxaLevel::Species => Color::from_rgb(255, 0, 0),    // Bright Red
            TaxaLevel::Subspecies => Color::from_rgb(200, 0, 0), // Darker Red
            TaxaLevel::Clade => Color::from_rgb(128, 0, 128),    // Purple
            TaxaLevel::Forma => Color::from_rgb(255, 20, 147),   // Deep Pink
            TaxaLevel::Varietas => Color::from_rgb(255, 20, 147), // Deep Pink
            TaxaLevel::SpeciesSubgroup => Color::from_rgb(200, 0, 0), // Darker Red
            TaxaLevel::Subcohort => Color::from_rgb(200, 50, 0), // Darker Red-Orange
            TaxaLevel::Cohort => Color::from_rgb(255, 69, 0),    // Red-Orange
            TaxaLevel::Section => Color::from_rgb(34, 139, 34),  // Forest Green
            TaxaLevel::Subsection => Color::from_rgb(34, 139, 34), // Forest Green
            TaxaLevel::Series => Color::from_rgb(34, 139, 34),   // Forest Green
        }
    }
}

pub type TaxonomyGraph = Graph<Taxon, (), Directed, u32>;

#[derive(Hash, Eq, PartialEq, Debug, Clone)]
pub struct Taxon {
    pub tax_id: u32,
    pub parent: u32,
    pub rank: TaxaLevel,
    pub name: String,
}

pub fn build_taxonomy_graph(
    nodes_file: &str,
    names_file: &str,
) -> (TaxonomyGraph, NodeIndex) {

    let taxa_names = parse_names(names_file.to_string());
    let nodes = parse_nodes(nodes_file.to_string());

    log::info!("Parsed names: {} total", taxa_names.len());
    // log::debug!("Parsed names (first 10): {:?}", &taxa_names[0..10]);
    log::debug!("Parsed names (u32, first 10): {:?}", &taxa_names.iter().map(|x| x.0).collect::<Vec<_>>()[0..10]);
    log::info!("Parsed nodes: {} total", nodes.0.len());
    log::debug!("Parsed nodes (first 10): {:?}", &nodes.0[0..10]);
    log::debug!("Parsed nodes (Strings, first 10): {:?}", &nodes.1[0..10]);

    let taxa_parents: HashMap<u32, u32> = nodes.0.iter().map(|(x, y)| (*x, *y)).collect();
    let taxa_ranks: HashMap<u32, String> = nodes.1.iter().zip(nodes.0.iter()).map(|(x, y)| (y.0, x.clone())).collect();

    assert!(taxa_names.get(&271808).unwrap() == "Wajira", "Taxa name is not correct: {}", taxa_names.get(&271808).unwrap());

    let edges: Vec<(u32, u32)> = nodes.0;

    let nodes: Vec<u32> = edges.iter().flat_map(|(x, y)| [x, y]).copied().collect();
    let nodes = nodes.into_iter().collect::<HashSet<u32>>();
    let nodes = nodes.into_iter().collect::<Vec<u32>>();

    log::debug!("Total Nodes: {}", nodes.len());
    log::debug!("Total Taxa Names: {}", taxa_names.len());
    assert!(nodes.len() == taxa_names.len(), "Nodes and names are not the same length");

    log::debug!("Total Edges: {}", edges.len());

    log::info!("Building taxonomy graph");

    // let mut graph = DiGraph::<u32, (), u32>::with_capacity(nodes.len(), nodes.len());
    let mut graph = DiGraph::<Taxon, (), u32>::with_capacity(nodes.len(), nodes.len());

    let nodes = nodes.into_iter().map(|x|
        Taxon {
            tax_id: x,
            parent: taxa_parents[&x],
            rank: TaxaLevel::from_str(&taxa_ranks[&x]),
            name: taxa_names[&x].clone(),
        }
    ).collect::<Vec<_>>();

    let nodes: HashMap<u32, NodeIndex> = nodes.into_iter().map(|x| (x.tax_id, graph.add_node(x))).collect();
    let node_indices: HashMap<NodeIndex, u32> = nodes.iter().map(|(x, y)| (*y, *x)).collect();

    let root = nodes[&1];

    for (x, y) in edges {
        graph.add_edge(nodes[&y], nodes[&x], ());
    }

    // Remove any that have no neighbors
    let mut removed = 0;
    for (i, ni) in nodes.iter() {
        let neighbors = graph.neighbors_undirected(*ni);
        if neighbors.count() == 0 {
            graph.remove_node(*ni);
            removed += 1;
        }
    }

    if removed > 0 {
        log::info!("Removed {} nodes with no neighbors", removed);
    }

    log::info!(
        "Taxonomy graph built. Node Count: {} - Edge Count: {}",
        graph.node_count(),
        graph.edge_count()
    );

    (graph, root)

}

// Store the taxa dist training element from each node in the graph, and update when given an new element
// Allows for training to continue even if new data is not yet ready
pub struct TaxaDistCache<const D: usize> {
    pub cache: Arc<RwLock<Vec<Option<TaxaDistance<D>>>>>,
    pub last_used: Arc<AtomicUsize>,
    pub highest_entry: Arc<AtomicUsize>,
    pub rx: Arc<crossbeam::channel::Receiver<(usize, TaxaDistance<D>)>>,
    pub tx: Arc<crossbeam::channel::Sender<(usize, TaxaDistance<D>)>>,

}

impl<const D: usize> TaxaDistCache<D> {
    pub fn with_capacity(capacity: usize) -> Self {

        let (tx, rx) = unbounded();

        Self {
            cache: Arc::new(RwLock::new(vec![None; capacity])),
            last_used: Arc::new(AtomicUsize::new(0)),
            highest_entry: Arc::new(AtomicUsize::new(0)),
            rx: Arc::new(rx),
            tx: Arc::new(tx),
        }
    }

    pub fn update(&self, i: usize, element: TaxaDistance<D>) {
        self.tx.send((i, element)).unwrap();

        if self.rx.len() > 8192 * 2 {
            let mut highest = self.highest_entry.load(std::sync::atomic::Ordering::Relaxed);
            let mut cache = self.cache.write().unwrap();
            while let Ok((i, element)) = self.rx.try_recv() {
                highest = highest.max(i);
                cache[i] = Some(element);

                if i == 0 {
                    log::debug!("Starting to fill again");
                }
            }

            self.highest_entry.store(highest, std::sync::atomic::Ordering::Relaxed);
        }        
    }

    pub fn len(&self) -> usize {
        self.highest_entry.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn get(&self, i: usize) -> TaxaDistance<D> {
        let cache = self.cache.read().unwrap();

        if let Some(x) = &cache[i] {
            return x.clone()
        }

        let mut val = None;
        let mut last_used = self.last_used.load(std::sync::atomic::Ordering::Relaxed);
        let mut highest = self.highest_entry.load(std::sync::atomic::Ordering::Relaxed);

        while val.is_none() {
            if let Some(x) = &cache[last_used] {
                val = Some(x.clone());
            } else {
                last_used += 1;
                if last_used >= highest {
                    last_used = 0;
                }
            }
        }

        self.last_used.store(last_used, std::sync::atomic::Ordering::Relaxed);

        val.unwrap()        
    }
}

pub fn build_taxonomy_graph_generator<const D: usize>(
    nodes_file: &str,
    names_file: &str,
    threads: usize,
) -> BatchGenerator<D> {
    let (graph, root) = build_taxonomy_graph(nodes_file, names_file);

    let (tx, rx) = bounded(threads * 8);

    let taxa_dist_cache: Arc<TaxaDistCache<D>> = Arc::new(TaxaDistCache::with_capacity(graph.node_count()));

    // Spawn threads
    let mut jhs = Vec::with_capacity(threads);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337);
    let graph = Arc::new(graph);
    let mut shutdown = Arc::new(AtomicBool::new(false));
    let current = Arc::new(AtomicUsize::new(0));

    let mut all_nodes = Arc::new(RwLock::new(Vec::new()));

    all_nodes = Arc::new(RwLock::new(graph.node_indices().collect::<Vec<_>>()));

    for threadno in 0..threads {
        let tx = tx.clone();
        let graph = Arc::clone(&graph);
        let shutdown = Arc::clone(&shutdown);
        rng.long_jump();
        let mut rng = rng.clone();
        let current = Arc::clone(&current);
        let all_nodes = Arc::clone(&all_nodes);
        let taxa_dist_cache = Arc::clone(&taxa_dist_cache);
        let root = root.clone();

        let jh = std::thread::spawn(move || {
            let mut local_excluded = HashSet::default();
            let mut branches = [0; D];
            let mut distances = [0; D];

            loop {
                if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }

                let idx = loop {
                    let i = current.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= all_nodes.read().unwrap().len() {
                        current.store(0, std::sync::atomic::Ordering::Relaxed);
                        continue;
                    }
                    break all_nodes.read().unwrap()[i];
                };

                local_excluded.clear();
                local_excluded.insert(idx);

                let nodes = random_walk_bfs(
                    Arc::as_ref(&graph),
                    &mut rng,
                    idx,
                    8,
                    &local_excluded,
                    D,
                );

                for (n, (ni, i)) in nodes.unwrap().iter().enumerate() {
                    branches[n] = ni.index() as u32;
                    distances[n] = *i;
                }

                // Far away samples
                // todo this calculates distance to root for the origin for each distant node,
                // this is inefficient
                let all_nodes_read = all_nodes.read().unwrap();
                for i in D / 2..D {
                    let end = *all_nodes_read.choose(&mut rng).unwrap();
                    // if let Some(distance) = dfs_distance_alt_taxalevel(Arc::as_ref(&graph), idx, end, &node_indices_levels) {
                    if let Some(distance) = root_calc(Arc::as_ref(&graph), idx, end, root) {
                        branches[i] = end.index() as u32;
                        distances[i] = distance as u32;
                    } else {
                        // If path not found, try another random node
                        continue;
                    }
                }

                // None of the distances should be 0
                if distances.iter().any(|&x| x == 0) {
                    // If so, show all branches and distances, and the origin
                    log::debug!(
                        "Branches: {:?}, Distances: {:?}, Origin: {}",
                        branches,
                        distances,
                        idx.index()
                    );
                    panic!();
                }

                let taxa_dist = TaxaDistance {
                    origin: idx.index() as u32,
                    branches,
                    distances,
                };

                // let names = taxa_dist.branches.iter().map(|x| graph.raw_nodes()[*x as usize].weight.name.clone()).collect::<Vec<_>>();
                // let branches_debug = taxa_dist.branches.iter().map(|x| graph.raw_nodes()[*x as usize].weight.tax_id).collect::<Vec<_>>();
                // let origin_tax_id = graph.raw_nodes()[idx.index() as usize].weight.tax_id;
                // let tax_ids_debug = taxa_dist.branches.iter().map(|x| graph.raw_nodes()[*x as usize].weight.tax_id).collect::<Vec<_>>();
                // log::debug!("Origin: {} {} - Branches TaxID: {:?} - Distances: {:?} - Names: {:?} - Tax IDs: {:?}", graph.raw_nodes()[taxa_dist.origin as usize].weight.name.clone(), origin_tax_id, branches_debug, taxa_dist.distances, names, tax_ids_debug);

                taxa_dist_cache.update(idx.index(), taxa_dist.clone());
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
        receiver: rx,
        taxa_dist_cache,
    }
}

fn random_walk_bfs<R: Rng>(
    graph: &TaxonomyGraph,
    rng: &mut R,
    start: NodeIndex,
    depth: usize,
    excluded_nodes: &HashSet<NodeIndex>,
    n: usize, // How many to return
) -> Option<Vec<(NodeIndex, u32)>> {
    let mut current_depth;

    let mut candidates = Vec::new();
    let mut results = Vec::new();

    for neighbor in graph.neighbors_undirected(start) {
        candidates.push((neighbor, 1_u32));
    }

    while let Some(node) = candidates.pop() {
        current_depth = node.1;

        // Stop early once we have passed the desired depth
        if current_depth as usize > depth {
            break;
        }

        for neighbor in graph.neighbors_undirected(node.0) {
            candidates.push((neighbor, (current_depth + 1) as u32));
        }
    }

    // Remove excluded nodes
    let filtered_candidates = candidates
        .iter()
        .filter(|x| !excluded_nodes.contains(&x.0))
        .collect::<Vec<_>>();

    if !filtered_candidates.is_empty() {
        while results.len() < n {
            // Weight by depth
            let dist = WeightedIndex::new(filtered_candidates.iter().map(|x| x.1)).unwrap();
            let next_node = filtered_candidates[dist.sample(rng)];
            results.push(*next_node);
        }
    } else {
        panic!();
    }

    Some(results)
}

fn random_walk<R: Rng>(
    graph: &TaxonomyGraph,
    rng: &mut R,
    start: NodeIndex,
    depth: usize,
    excluded_nodes: &HashSet<NodeIndex>,
) -> Option<(NodeIndex, u32)> {
    let mut current_node = start;
    let mut visited_nodes = vec![current_node];

    for curdepth in 1..depth {
        let mut neighbors: Vec<_> = graph
            .neighbors_undirected(current_node)
            .filter(|&n| !visited_nodes.contains(&n))
            .collect();

        if depth == 1 && neighbors.is_empty() {
            log::debug!("Node {} has no neighbors", current_node.index());
        }

        if neighbors.is_empty() {
            if start != current_node {
                return Some((current_node, curdepth as u32));
            } else {
                return None;
            }
        }

        neighbors = neighbors
            .into_iter()
            .filter(|x| !excluded_nodes.contains(x))
            .collect();

        if neighbors.is_empty() {
            if start != current_node {
                return Some((current_node, curdepth as u32));
            } else {
                return None;
            }
        }

        /*        let mut neighbors_neighbor_count = neighbors
        .iter()
        .map(|x| graph.neighbors(*x).count())
        .collect::<Vec<_>>(); */

        // if curdepth < depth - 1 {
        // Means we have more than one level to go, so try to avoid dead ends
        // Filter neighbors that have more than 1 neighbor
        // And filter neighbor counts to use as weights

        /*
        neighbors = neighbors_neighbor_count
            .iter()
            .enumerate()
            .filter(|(_, x)| **x > 1)
            .map(|(i, _)| neighbors[i])
            .collect();

        neighbors_neighbor_count = neighbors_neighbor_count
            .into_iter()
            .filter(|x| *x > 1)
            .collect();
        */
        // }

        if neighbors.is_empty() {
            if start != current_node {
                return Some((current_node, curdepth as u32));
            } else {
                return None;
            }
        }

        // let dist = WeightedIndex::new(neighbors_neighbor_count).unwrap();

        // let next_node = neighbors[dist.sample(&mut rng)];

        let mut next_node = rng.gen_range(0..neighbors.len());
        while curdepth < depth - 1 && graph.neighbors(neighbors[next_node]).count() == 1 {
            if neighbors.len() == 1 {
                current_node = neighbors[next_node];
                if start != current_node {
                    return Some((current_node, curdepth as u32));
                } else {
                    return None;
                }
            }

            // Remove from neighbors
            neighbors.remove(next_node);

            next_node = rng.gen_range(0..neighbors.len());
        }

        visited_nodes.push(neighbors[next_node]);
        current_node = neighbors[next_node];
    }

    Some((current_node, depth as u32))
}

fn random_walk_alt<R: Rng>(
    graph: &TaxonomyGraph,
    rng: &mut R,
    start: NodeIndex,
    depth: usize,
    excluded_nodes: &HashSet<NodeIndex>,
) -> Option<(NodeIndex, u32)> {
    let mut current_node = start;
    let mut visited_nodes = HashSet::default();
    visited_nodes.insert(current_node);

    for current_depth in 0..depth {
        let neighbors: Vec<_> = graph
            .neighbors(current_node)
            .filter(|&n| !visited_nodes.contains(&n) && !excluded_nodes.contains(&n))
            .collect();

        if neighbors.is_empty() {
            return if current_node != start {
                Some((current_node, current_depth as u32))
            } else {
                None
            };
        }

        current_node = *neighbors.choose(rng).unwrap();
        visited_nodes.insert(current_node);
    }

    Some((current_node, depth as u32))
}

fn bfs_distance(
    graph: &TaxonomyGraph,
    start: NodeIndex,
    end: NodeIndex,
) -> Option<usize> {
    // Initialize BFS
    let mut bfs = Bfs::new(graph, start);
    let mut distances = vec![None; graph.node_count()];
    distances[start.index()] = Some(0);

    // Perform BFS
    while let Some(node) = bfs.next(graph) {
        let current_distance = distances[node.index()].unwrap();

        if node == end {
            return Some(current_distance);
        }

        for neighbor in graph.neighbors(node) {
            if distances[neighbor.index()].is_none() {
                distances[neighbor.index()] = Some(current_distance + 1);
            }
        }
    }

    // If we finish BFS without finding the target node, return None
    None
}

fn bfs_distance_alt(
    graph: &TaxonomyGraph,
    start: NodeIndex,
    end: NodeIndex,
    max_depth: Option<usize>,
) -> Option<usize> {
    let mut queue = VecDeque::new();
    let mut visited = HashSet::default();

    queue.push_back((start, 0));
    visited.insert(start);

    while let Some((node, distance)) = queue.pop_front() {
        if node == end {
            return Some(distance);
        }

        if let Some(max) = max_depth {
            if distance >= max {
                continue;
            }
        }

        for neighbor in graph.neighbors(node) {
            if visited.insert(neighbor) {
                queue.push_back((neighbor, distance + 1));
            }
        }
    }

    None
}

fn dfs_distance(
    graph: &TaxonomyGraph,
    start: NodeIndex,
    end: NodeIndex,
) -> Option<usize> {
    // Initialize BFS
    let mut bfs = Dfs::new(graph, start);
    let mut distances = vec![None; graph.node_count()];
    distances[start.index()] = Some(0);

    // Perform DFS
    while let Some(node) = bfs.next(graph) {
        let current_distance = distances[node.index()].unwrap();

        if node == end {
            return Some(current_distance);
        }

        for neighbor in graph.neighbors(node) {
            if distances[neighbor.index()].is_none() {
                distances[neighbor.index()] = Some(current_distance + 1);
            }
        }
    }

    // If we finish DFS without finding the target node, return None
    None
}

fn dfs_distance_alt(
    graph: &TaxonomyGraph,
    start: NodeIndex,
    end: NodeIndex,
) -> Option<usize> {
    let mut queue = VecDeque::new();
    let mut visited = HashSet::default();
    queue.push_back((start, 0));
    visited.insert(start);

    while let Some((node, distance)) = queue.pop_front() {
        if node == end {
            return Some(distance);
        }

        for neighbor in graph.neighbors(node) {
            if visited.insert(neighbor) {
                queue.push_back((neighbor, distance + 1));
            }
        }
    }

    None
}

// Taxa level aware
fn dfs_distance_alt_taxalevel(
    graph: &TaxonomyGraph,
    start: NodeIndex,
    end: NodeIndex,
    node_indices_levels: &Arc<HashMap<NodeIndex, TaxaLevel>>,
) -> Option<usize> {
    let mut queue = VecDeque::new();
    let mut visited = HashSet::default();
    queue.push_back((start, 0));
    visited.insert(start);

    let target_taxa_level = node_indices_levels[&end];
    let start_taxa_level = node_indices_levels[&start];

    let no_deeper_than = target_taxa_level.max(start_taxa_level);

    log::debug!(
        "Start: {:?} - End: {:?} - Target Level: {:?} - Start Level: {:?} - No Deeper Than: {:?}",
        start,
        end,
        target_taxa_level,
        start_taxa_level,
        no_deeper_than
    );



    while let Some((node, distance)) = queue.pop_front() {
        if node == end {
            return Some(distance);
        }

        for neighbor in graph.neighbors(node) {
            let neighbor_taxa_level = node_indices_levels[&neighbor];

            // If neighbor is deeper than target AND the starting node, skip
            if neighbor_taxa_level <= no_deeper_than && visited.insert(neighbor) {
                queue.push_back((neighbor, distance + 1));
            }
        }
    }

    None
}

// Taxa level aware
fn root_calc(
    graph: &TaxonomyGraph,
    start: NodeIndex,
    end: NodeIndex,
    root: NodeIndex,
) -> Option<usize> {
    let mut cur_node = start;
    let mut path = vec![cur_node];
    while cur_node != root {
        let edges = graph.edges_directed(cur_node, Incoming);

        // Make sure there is only one parent
        let edges = edges.collect::<Vec<_>>();
        if edges.len() > 1 {
            panic!("More than one parent");
        }

        if edges.len() == 0 {
            log::debug!("No Parent: {} {} {}",
                cur_node.index(),
                start.index(),
                end.index()
            );
            panic!("No parent");
        }

        let parent = edges[0].source();
        path.push(parent);
        cur_node = parent;
    }

    let mut path2 = vec![end];
    let mut cur_node = end;
    while cur_node != root {
        let edges = graph.edges_directed(cur_node, Incoming);
        // Make sure there is only one parent
        let edges = edges.collect::<Vec<_>>();
        if edges.len() > 1 {
            panic!("More than one parent");
        }

        let parent = edges[0].source();
        path2.push(parent);
        cur_node = parent;
    }

    let mut common = 0;
    for (i, node) in path.iter().enumerate() {
        if path2.contains(node) {
            common = i;
            break;
        }
    }

    let mut common2 = 0;
    for (i, node) in path2.iter().enumerate() {
        if path.contains(node) {
            common2 = i;
            break;
        }
    }

    Some(path.len() + path2.len() +1 - common + common2)
}

pub struct BatchGenerator<const D: usize> {
    pub root: NodeIndex,
    pub graph: Arc<Graph<Taxon, (), Directed, u32>>,
    epoch_size: usize,
    join_handles: Vec<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    receiver: crossbeam::channel::Receiver<TaxaDistance<D>>,
    pub taxa_dist_cache: Arc<TaxaDistCache<D>>,
}

impl<const D: usize> Dataset<TaxaDistance<D>> for BatchGenerator<D> {
    fn len(&self) -> usize {
        self.epoch_size
    }

    fn get(&self, index: usize) -> Option<TaxaDistance<D>> {
        // self.receiver.recv().ok()
        self.taxa_dist_cache.get(index).into()
    }

    // Provided methods
    fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    fn iter(&self) -> DatasetIterator<'_, TaxaDistance<D>>
    where
        Self: Sized,
    {
        DatasetIterator::new(self)
    }
}

impl<const D: usize> BatchGenerator<D> {

    pub fn precache(&self) {

        // Optimal is 10%
        let cache_warmup_size = self.graph.node_count() / 10;

        // In development mode is less though...
        // let cache_warmup_size = 2048;

        while self.taxa_dist_cache.len() < cache_warmup_size {
            println!(
                "Waiting for cache to fill: {}/{}",
                self.taxa_dist_cache.len(),
                cache_warmup_size
            );
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
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
            receiver: self.receiver.clone(),
            taxa_dist_cache: Arc::clone(&self.taxa_dist_cache),
        }
    }

    pub fn taxonomy_size(&self) -> usize {
        self.graph.node_count()
    }
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
