use bumpalo::Bump;
use burn::data::dataset::{Dataset, DatasetIterator};
use crossbeam::channel::bounded;
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

#[derive(PartialEq, PartialOrd, Eq, Ord, Debug, Clone, Copy)]
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

pub fn build_taxonomy_graph(
    nodes_file: &str,
    names_file: &str,
) -> (
    petgraph::Graph<u32, (), petgraph::Undirected>,
    HashMap<NodeIndex, u32>,
    Vec<String>,
    HashMap<u32, TaxaLevel>,
    Vec<String>,
    HashMap<u32, NodeIndex>,
    Vec<Color>,
) {
    let names = parse_names(names_file.to_string());
    let nodes = parse_nodes(nodes_file.to_string());

    log::info!("Parsed names: {} total", names.0.len());
    log::debug!("Parsed names (first 10): {:?}", &names.0[0..10]);
    log::debug!("Parsed names (u32, first 10): {:?}", &names.1[0..10]);
    log::info!("Parsed nodes: {} total", nodes.0.len());
    log::debug!("Parsed nodes (first 10): {:?}", &nodes.0[0..10]);
    log::debug!("Parsed nodes (Strings, first 10): {:?}", &nodes.1[0..10]);

    // When a parent is 0 it means drop it, 1 is the root
    let edges: Vec<(u32, u32)> = nodes.0;

    let levels: HashMap<u32, TaxaLevel> = edges
        .iter()
        .zip(nodes.1.iter())
        .map(|((tax_id, _parent), rank)| {
            let rank = TaxaLevel::from_str(&rank);
            (*tax_id, rank)
        })
        .collect();

    let levels_in_order: Vec<String> = nodes.1.iter().map(|x| x.clone()).collect();

    let colors: Vec<Color> = nodes
        .1
        .iter()
        .map(|rank| TaxaLevel::from_str(rank).color())
        .collect();

    let taxa_names: Vec<String> = names.0;

    let nodes: Vec<u32> = edges.iter().flat_map(|(x, y)| [x, y]).copied().collect();
    let nodes = nodes.into_iter().collect::<HashSet<u32>>();
    let nodes = nodes.into_iter().collect::<Vec<u32>>();

    log::debug!("Total Edges: {}", edges.len());

    log::info!("Building taxonomy graph");

    let mut graph = UnGraph::<u32, (), u32>::with_capacity(nodes.len(), nodes.len());
    let nodes: HashMap<u32, NodeIndex> = nodes.iter().map(|x| (*x, graph.add_node(1))).collect();
    let node_indices: HashMap<NodeIndex, u32> = nodes.iter().map(|(x, y)| (*y, *x)).collect();

    for (x, y) in edges {
        graph.add_edge(nodes[&x], nodes[&y], ());
    }

    // Remove any that have no neighbors
    let mut removed = 0;
    for (i, ni) in nodes.iter() {
        let neighbors = graph.neighbors(*ni);
        if neighbors.count() == 0 {
            graph.remove_node(*ni);
            removed += 1;
        }
    }

    log::info!("Removed {} nodes with no neighbors", removed);

    // Rebuild nodes and node_indices
    let nodes = graph
        .node_indices()
        .map(|x| (node_indices[&x], x))
        .collect::<HashMap<_, _>>();
    let node_indices = nodes
        .iter()
        .map(|(x, y)| (*y, *x))
        .collect::<HashMap<_, _>>();

    log::info!(
        "Taxonomy graph built. Node Count: {} - Edge Count: {}",
        graph.node_count(),
        graph.edge_count()
    );

    (
        graph,
        node_indices,
        taxa_names,
        levels,
        levels_in_order,
        nodes,
        colors,
    )
}

// Store the taxa dist training element from each node in the graph, and update when given an new element
// Allows for training to continue even if new data is not yet ready
pub struct TaxaDistCache<const D: usize> {
    pub cache: Arc<RwLock<Vec<TaxaDistance<D>>>>,
    pub last_used: Arc<AtomicUsize>,
}

impl<const D: usize> TaxaDistCache<D> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(Vec::with_capacity(capacity))),
            last_used: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn update(&self, i: usize, element: TaxaDistance<D>) {
        let mut cache = self.cache.write().unwrap();
        if cache.len() <= i {
            cache.resize(i + 1, element);
        } else {
            cache[i] = element;
        }
    }

    pub fn len(&self) -> usize {
        self.cache.read().unwrap().len()
    }

    pub fn get(&self, i: usize) -> TaxaDistance<D> {
        let cache = self.cache.read().unwrap();

        if i >= cache.len() {
            if self.last_used.load(std::sync::atomic::Ordering::Relaxed) >= cache.len() {
                self.last_used
                    .store(0, std::sync::atomic::Ordering::Relaxed);
            }

            cache[self
                .last_used
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)]
            .clone()
        } else {
            cache[i].clone()
        }
    }
}

pub fn build_taxonomy_graph_generator<const D: usize>(
    nodes_file: &str,
    names_file: &str,
    threads: usize,
) -> BatchGenerator<D> {
    let (graph, node_indices, taxa_names, levels, levels_in_order, nodes, colors) =
        build_taxonomy_graph(nodes_file, names_file);

    let (tx, rx) = bounded(threads * 2);

    let taxa_dist_cache: Arc<TaxaDistCache<D>> =
        Arc::new(TaxaDistCache::with_capacity(graph.node_count()));

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

        let jh = std::thread::spawn(move || {
            let mut local_excluded = HashSet::default();
            let mut branches = [0; D];
            let mut distances = [0; D];
            let mut depths = vec![None; graph.node_count()];

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

                // Nearby samples
                for i in 0..D / 2 {
                    let mut depth = rng.gen_range(1..6);

                    let mut node = None;
                    let mut tries = 0;
                    while node.is_none() {
                        if tries > 10 {
                            depth += 1;
                        }

                        if tries > 1000 {
                            log::debug!("Node {} has no neighbors", idx.index());
                            break;
                        }

                        tries += 1;
                        node = random_walk_bfs(
                            Arc::as_ref(&graph),
                            &mut rng,
                            idx,
                            depth,
                            &local_excluded,
                            &mut depths,
                        );
                        depths.iter_mut().for_each(|x| *x = None);
                    }

                    let node = node.unwrap();

                    local_excluded.insert(node.0);
                    branches[i] = node.0.index() as u32;
                    distances[i] = node.1;
                }

                // Far away samples
                let all_nodes_read = all_nodes.read().unwrap();
                for i in D / 2..D {
                    let end = *all_nodes_read.choose(&mut rng).unwrap();
                    if let Some(distance) = dfs_distance_alt(Arc::as_ref(&graph), idx, end) {
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

                taxa_dist_cache.update(idx.index(), taxa_dist.clone());
            }
        });

        jhs.push(jh);
    }

    let cache_warmup_size = graph.node_count() / 10;
    let cache_warmup_size = 8192;

    while taxa_dist_cache.len() < cache_warmup_size {
        println!(
            "Waiting for cache to fill: {}/{}",
            taxa_dist_cache.len(),
            cache_warmup_size
        );
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    BatchGenerator {
        epoch_size: graph.node_count(),
        graph: graph,
        levels,
        join_handles: jhs,
        shutdown,
        receiver: rx,
        nodes,
        colors,
        levels_in_order,
        taxa_names,
        taxa_dist_cache,
    }
}

fn random_walk_bfs<R: Rng>(
    graph: &Graph<u32, (), petgraph::Undirected, u32>,
    rng: &mut R,
    start: NodeIndex,
    depth: usize,
    excluded_nodes: &HashSet<NodeIndex>,
    depths: &mut Vec<Option<usize>>,
) -> Option<(NodeIndex, u32)> {
    let mut bfs = Bfs::new(graph, start);
    let mut current_depth;

    depths[start.index()] = Some(0);

    let mut candidates = Vec::new();

    while let Some(node) = bfs.next(graph) {
        current_depth = depths[node.index()].unwrap_or(0);

        // Stop early once we have passed the desired depth
        if current_depth > depth {
            break;
        }

        for neighbor in graph.neighbors(node) {
            if depths[neighbor.index()].is_none() {
                depths[neighbor.index()] = Some(current_depth + 1);
                if current_depth + 1 == depth && !excluded_nodes.contains(&neighbor) {
                    candidates.push((neighbor, (current_depth + 1) as u32));
                }
            }
        }
    }

    if !candidates.is_empty() {
        Some(*candidates.choose(rng).unwrap())
    } else {
        None
    }
}

fn random_walk<R: Rng>(
    graph: &Graph<u32, (), Undirected, u32>,
    rng: &mut R,
    start: NodeIndex,
    depth: usize,
    excluded_nodes: &HashSet<NodeIndex>,
) -> Option<(NodeIndex, u32)> {
    let mut current_node = start;
    let mut visited_nodes = vec![current_node];

    for curdepth in 1..depth {
        let mut neighbors: Vec<_> = graph
            .neighbors(current_node)
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
    graph: &Graph<u32, (), Undirected, u32>,
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
    graph: &Graph<u32, (), Undirected, u32>,
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
    graph: &Graph<u32, (), Undirected, u32>,
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
    graph: &Graph<u32, (), Undirected, u32>,
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
    graph: &Graph<u32, (), Undirected, u32>,
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

pub struct BatchGenerator<const D: usize> {
    pub graph: Arc<Graph<u32, (), Undirected, u32>>,
    epoch_size: usize,
    pub levels: HashMap<u32, TaxaLevel>,
    join_handles: Vec<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    receiver: crossbeam::channel::Receiver<TaxaDistance<D>>,
    pub nodes: HashMap<u32, NodeIndex>,
    pub colors: Vec<Color>,
    pub levels_in_order: Vec<String>,
    pub taxa_names: Vec<String>,
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

    /*

    pub fn testing() -> Self {
        let mut graph = UnGraph::<u32, (), u32>::with_capacity(100_000, 100_000);
        let nodes: HashMap<u32, NodeIndex> = (0..100_000_u32)
            .into_iter()
            .map(|x| (x, graph.add_node(1)))
            .collect();

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337);

        // Connect nodes to each other in order
        for i in 0..100_000_u32 {
            if i > 0 {
                graph.add_edge(nodes[&(i - 1)], nodes[&i], ());
            }
        }

        let levels = HashMap::new();

        Self {
            graph: Arc::new(graph),
            rng,
            epoch_size: 1024,
            levels,
        }
    }
    */

    pub fn valid(&self) -> Self {
        BatchGenerator {
            graph: Arc::clone(&self.graph),
            epoch_size: 2048,
            levels: self.levels.clone(),
            join_handles: vec![],
            shutdown: Arc::clone(&self.shutdown),
            receiver: self.receiver.clone(),
            nodes: self.nodes.clone(),
            colors: self.colors.clone(),
            levels_in_order: self.levels_in_order.clone(),
            taxa_names: self.taxa_names.clone(),
            taxa_dist_cache: Arc::clone(&self.taxa_dist_cache),
        }
    }

    pub fn taxonomy_size(&self) -> usize {
        self.graph.node_count()
    }
}

pub fn parse_nodes(filename: String) -> (Vec<(u32, u32)>, Vec<String>) {
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

pub fn parse_names(filename: String) -> (Vec<String>, Vec<u32>) {
    let mut names: Vec<String> = Vec::with_capacity(3_006_098);

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
        let class: &str = &split[3];

        match names.get(id) {
            None => {
                names.resize(id + 1, "".to_string());
                names[id] = name.into();
                taxids.insert(id as u32);
            }
            Some(_) => {
                if class == "scientific name" {
                    names[id] = name.into();
                }
            }
        };
    }

    let mut taxids = taxids.into_iter().collect::<Vec<u32>>();
    taxids.sort_unstable();

    (names, taxids)
}
