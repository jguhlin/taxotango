use burn::data::dataset::{Dataset, DatasetIterator};
use petgraph::adj::NodeIndices;
use petgraph::algo::astar;
use petgraph::prelude::*;
use crossbeam::channel::bounded;
use petgraph::graph::{NodeIndex, UnGraph};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_xoshiro::Xoshiro256PlusPlus;
use rerun::Color;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;

pub mod model;
pub use model::*;

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

pub fn build_taxonomy_graph_generator<const D: usize>(
    nodes_file: &str,
    names_file: &str,
    threads: usize,
) -> BatchGenerator<D> {
    // acc2tax todo: change it so we are passing &str
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

    let levels: HashMap<u32, TaxaLevel> = edges.iter().zip(nodes.1.iter()).map(
        |((tax_id, _parent), rank)| {
            let rank = TaxaLevel::from_str(&rank);
            (*tax_id, rank)
        }
    ).collect();

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

    log::info!(
        "Taxonomy graph built. Node Count: {} - Edge Count: {}",
        graph.node_count(),
        graph.edge_count()
    );

    // Iterate through all nodes and calculate how many connections there are (median, max, min)
    let mut max = 0;
    let mut min = std::u32::MAX;
    let mut total = 0;

    let median = graph
        .node_indices()
        .map(|idx| graph.neighbors(idx).count() as u32)
        .collect::<Vec<u32>>();

    // How many are over 100, 1000, 10000
    let over_100 = median.iter().filter(|x| **x > 100).count();
    let over_1000 = median.iter().filter(|x| **x > 1000).count();
    let over_10000 = median.iter().filter(|x| **x > 10000).count();

    // over 200, 300, 400, 500
    let over_200 = median.iter().filter(|x| **x > 200).count();
    let over_300 = median.iter().filter(|x| **x > 300).count();
    let over_400 = median.iter().filter(|x| **x > 400).count();
    let over_500 = median.iter().filter(|x| **x > 500).count();

    for i in &median {
        total += i;
        if *i > max {
            max = *i;
        }

        if *i < min {
            min = *i;
        }
    }

    let median_pos = total / median.len() as u32;
    let median = median[median_pos as usize];

    log::info!("Median: {} - Max: {} - Min: {}", median, max, min);
    log::info!(
        "Over 100: {} - Over 1000: {} - Over 10000: {}",
        over_100,
        over_1000,
        over_10000
    );

    println!("Median: {} - Max: {} - Min: {}", median, max, min);
    println!(
        "Over 100: {} - Over 1000: {} - Over 10000: {}",
        over_100, over_1000, over_10000
    );
    println!(
        "Over 200: {} - Over 300: {} - Over 400: {} - Over 500: {}",
        over_200, over_300, over_400, over_500
    );

    // Remove all from the graph that have over 200 connections
    // todo make this smart
    /*
    let mut to_remove = vec![];

    for idx in graph.node_indices() {
        if graph.neighbors(idx).count() > 200 {
            to_remove.push(idx);
        }
    }

    for idx in to_remove {
        graph.remove_node(idx);
    } */

    if petgraph::algo::is_cyclic_undirected(&graph) {
        println!("Graph is cyclic");
    }

    /*
    // Testing stuff
    let mut neighbors = graph.neighbors(nodes[&262]).collect::<Vec<_>>();
    println!("Neighbors of 262: {:?}", neighbors);
    let converted_back = neighbors.iter().map(|x| node_indices[x]).collect::<Vec<_>>();
    println!("Neighbors of 262: {:?}", converted_back);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337);
    let output = random_walk(&graph, &mut rng, nodes[&262], 5, vec![]);
    println!("Random walk: {:?}", output);

    panic!(); */

    let (tx, rx) = bounded(8192 * 16);

    // Spawn threads
    let mut jhs = Vec::with_capacity(threads);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337);
    let graph = Arc::new(graph);
    let mut shutdown = Arc::new(AtomicBool::new(false));
    let current = Arc::new(AtomicUsize::new(0));

    let mut all_nodes = Arc::new(RwLock::new(Vec::new()));

    all_nodes = Arc::new(RwLock::new(graph.node_indices().collect::<Vec<_>>()));

    for _ in 0..threads {
        let tx = tx.clone();
        let graph = Arc::clone(&graph);
        let shutdown = Arc::clone(&shutdown);
        let rng1 = rng.clone();
        let current = Arc::clone(&current);
        let all_nodes = Arc::clone(&all_nodes);

        let jh = std::thread::spawn(move || {
            let mut rng = rng1;

            // 1/2 are nearby, 1/2 are far away as 'negative' samples
            let cutoff = (D as f32 * 0.5) as usize;
            let further = D - cutoff;

            'outer: loop {
                if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }

                // let i = rng.gen_range(0..len) as u32;
                // let idx = graph.node_indices().nth(i as usize).unwrap();
                let mut idx = current.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                if idx >= all_nodes.read().unwrap().len() {
                    current.store(0, std::sync::atomic::Ordering::Relaxed);
                    idx = current.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }

                let idx = all_nodes.read().unwrap()[idx];

                let mut branches = [0; D];
                let mut distances = [0; D];

                let mut used = HashSet::new();

                for i in 0..cutoff {
                    let depth = rng.gen_range(1..6);
                    let mut node = None;

                    let mut iters = 0;
                    while node.is_none() {
                        iters += 1;

                        if iters > 100 {
                            // Remove this node from the list
                            all_nodes.write().unwrap().retain(|&x| x != idx);
                            log::debug!("Removed node: {}", idx.index());
                            continue 'outer;
                        }

                        node = random_walk(
                            Arc::as_ref(&graph),
                            &mut rng,
                            idx,
                            depth,
                            vec![idx],
                        );
                    }

                    let node = node.unwrap();

                    // Get actual distance with astar

                    let distance = astar(
                        Arc::as_ref(&graph),
                        idx,
                        |finish| finish == node,
                        |_| 1,
                        |_| 0,
                    );

                    let distance = distance.unwrap().0 as u8;

                    if node == idx {
                        panic!("Same node");
                    }

                    used.insert(node);
                    branches[i] = node.index() as u32;
                    distances[i] = distance as u32;
                }

                for i in cutoff..further {
                    // Pick a completely random node
                    let end = all_nodes.read().unwrap().choose(&mut rng).unwrap().clone();
                        
                    let distance = astar(
                        Arc::as_ref(&graph),
                        idx,
                        |finish| finish == end,
                        |_| 1,
                        |_| 0,
                    );

                    // let (node, depth) = self.random_walk(idx, depth);
                    branches[i] = end.index() as u32;
                    distances[i] = distance.unwrap().0 as u32;
                }

                tx.send(TaxaDistance {
                    origin: idx.index() as u32,
                    branches,
                    distances,
                })
                .unwrap();
            }
        });

        rng.long_jump();
        jhs.push(jh);
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
    }
}

fn random_walk<R: Rng>(
    graph: &Graph<u32, (), Undirected, u32>,
    mut rng: &mut R,
    start: NodeIndex,
    depth: usize,
    excluded_nodes: Vec<NodeIndex>,
) -> Option<NodeIndex> {
    let mut current_node = start;
    let mut visited_nodes = vec![current_node];

    for curdepth in 0..depth {
        let mut neighbors: Vec<_> = graph
            .neighbors(current_node)
            .filter(|&n| !visited_nodes.contains(&n))
            .collect();

        if neighbors.is_empty() {
            if start != current_node {
                return Some(current_node);
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
                return Some(current_node);
            } else {
                return None;
            }
        }

        let mut neighbors_neighbor_count = neighbors
            .iter()
            .map(|x| graph.neighbors(*x).count())
            .collect::<Vec<_>>();

        if curdepth < depth - 1 {
            // Means we have more than one level to go, so try to avoid dead ends
            // Filter neighbors that have more than 1 neighbor
            // And filter neighbor counts to use as weights
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
        }

        if neighbors.is_empty() {
            if start != current_node {
                return Some(current_node);
            } else {
                return None;
            }
        }

        let dist = WeightedIndex::new(neighbors_neighbor_count).unwrap();

        let next_node = neighbors[dist.sample(&mut rng)];
        visited_nodes.push(next_node);
        current_node = next_node;
    }
    

    Some(current_node)
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
}

impl<const D: usize> Dataset<TaxaDistance<D>> for BatchGenerator<D> {
    fn len(&self) -> usize {
        // Batch size of 1024
        self.epoch_size
    }

    fn get(&self, _index: usize) -> Option<TaxaDistance<D>> {
        self.receiver.recv().ok()
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
        let rng = Xoshiro256PlusPlus::seed_from_u64(13371337);
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
    let mut taxids = HashSet::new();

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
