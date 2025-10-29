import os
import pandas as pd
from whoosh import index
from whoosh.qparser import QueryParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


class InformationRetrievalSystem:
    def __init__(self):
        self.themes = ['etd_ugm_clean', 'etd_usk_clean', 'kompas_clean', 'mojok_clean_all', 'tempo_clean']
        self.main_index_dir = "index"
        self.dataset_dir = "dataset_clean"
        self.indices = {}
        self.documents = {}
        self.vectorizers = {}
        self.doc_vectors = {}
        
        # Initialize Sastrawi
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        
    def preprocess_query(self, query):
        """
        Preprocessing query sama seperti preprocessing dokumen
        """
        # Case folding
        query = query.lower()
        
        # Remove special characters and numbers
        query = re.sub(r'[^a-z\s]', '', query)
        
        # Remove stopwords
        query = self.stopword_remover.remove(query)
        
        # Stemming
        query = self.stemmer.stem(query)
        
        # Remove extra spaces
        query = ' '.join(query.split())
        
        return query
    
    def load_and_index_datasets(self):
        """
        Load datasets dan index yang sudah ada
        """
        print("\n🔄 Loading datasets and indices...")
        
        for theme in self.themes:
            try:
                # Load CSV data
                csv_path = os.path.join(self.dataset_dir, f'{theme}.csv')
                df = pd.read_csv(csv_path)
                
                # Simpan dokumen
                self.documents[theme] = df
                
                # Load Whoosh index
                theme_index_dir = os.path.join(self.main_index_dir, f'index_{theme}')
                if os.path.exists(theme_index_dir):
                    self.indices[theme] = index.open_dir(theme_index_dir)
                else:
                    print(f"⚠️  Index untuk {theme} tidak ditemukan. Silakan buat index terlebih dahulu.")
                    continue
                
                # Buat vectorizer dan document vectors untuk cosine similarity
                texts = df['clean_tokens'].astype(str).tolist()
                
                # Convert string representation of list to actual text
                processed_texts = []
                for text in texts:
                    # Remove brackets and quotes, replace commas with spaces
                    cleaned = text.strip("[]").replace("'", "").replace(",", " ")
                    processed_texts.append(cleaned)
                
                vectorizer = CountVectorizer()
                doc_vectors = vectorizer.fit_transform(processed_texts)
                
                self.vectorizers[theme] = vectorizer
                self.doc_vectors[theme] = doc_vectors
                
                print(f"✅ Loaded {theme}: {len(df)} documents")
                
            except Exception as e:
                print(f"❌ Error loading {theme}: {str(e)}")
        
        print(f"\n✅ Successfully loaded {len(self.documents)} datasets")
        print(f"📊 Total documents: {sum(len(df) for df in self.documents.values())}")
    
    def search_whoosh(self, query_text, theme, limit=10):
        """
        Search menggunakan Whoosh index
        """
        if theme not in self.indices:
            return []
        
        ix = self.indices[theme]
        results = []
        
        with ix.searcher() as searcher:
            query = QueryParser("content", ix.schema).parse(query_text)
            search_results = searcher.search(query, limit=limit)
            
            for hit in search_results:
                results.append({
                    'id': hit['id'],
                    'content': hit['content'],
                    'score': hit.score
                })
        
        return results
    
    def calculate_cosine_similarity(self, query_text, theme, top_n=5):
        """
        Calculate cosine similarity between query and documents
        """
        if theme not in self.vectorizers or theme not in self.doc_vectors:
            return []
        
        # Preprocess query
        processed_query = self.preprocess_query(query_text)
        
        # Transform query using the same vectorizer
        try:
            query_vector = self.vectorizers[theme].transform([processed_query])
        except Exception as e:
            print(f"⚠️  Error transforming query: {str(e)}")
            return []
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.doc_vectors[theme])[0]
        
        # Get top N documents
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        results = []
        df = self.documents[theme]
        
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include documents with similarity > 0
                result = {
                    'rank': len(results) + 1,
                    'doc_id': idx,
                    'theme': theme,
                    'similarity_score': float(similarities[idx]),
                    'content': df.iloc[idx]['clean_tokens'] if idx < len(df) else "N/A"
                }
                results.append(result)
        
        return results
    
    def search_all_datasets(self, query_text, top_n=5):
        """
        Search across all datasets and combine results
        """
        all_results = []
        
        print(f"\n🔍 Searching for: '{query_text}'")
        print(f"📝 Preprocessed query: '{self.preprocess_query(query_text)}'")
        print("\n" + "="*70)
        
        for theme in self.themes:
            results = self.calculate_cosine_similarity(query_text, theme, top_n)
            all_results.extend(results)
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top N results
        return all_results[:top_n]
    
    def search_single_dataset(self, query_text, theme, top_n=5):
        """
        Search in a single dataset
        """
        if theme not in self.documents:
            print(f"❌ Dataset {theme} tidak ditemukan!")
            return []
        
        print(f"\n🔍 Searching in {theme} for: '{query_text}'")
        print(f"📝 Preprocessed query: '{self.preprocess_query(query_text)}'")
        print("\n" + "="*70)
        
        results = self.calculate_cosine_similarity(query_text, theme, top_n)
        return results
    
    def display_results(self, results):
        """
        Display search results in a formatted way
        """
        if not results:
            print("\n❌ No results found.")
            return
        
        print(f"\n📊 Found {len(results)} relevant documents:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{'='*70}")
            print(f"Rank #{i}")
            print(f"Dataset: {result['theme']}")
            print(f"Document ID: {result['doc_id']}")
            print(f"Similarity Score: {result['similarity_score']:.4f}")
            print(f"Content Preview: {str(result['content'])[:200]}...")
            print()
    
    def show_menu(self):
        """
        Display main menu
        """
        print("\n" + "="*70)
        print(" " * 15 + "INFORMATION RETRIEVAL SYSTEM")
        print("="*70)
        print("[1] Load & Index Dataset")
        print("[2] Search Query - All Datasets")
        print("[3] Search Query - Specific Dataset")
        print("[4] Show Dataset Statistics")
        print("[5] Exit")
        print("="*70)
    
    def show_dataset_menu(self):
        """
        Display dataset selection menu
        """
        print("\n📚 Available Datasets:")
        for i, theme in enumerate(self.themes, 1):
            print(f"[{i}] {theme}")
        print(f"[{len(self.themes) + 1}] Back to Main Menu")
    
    def show_statistics(self):
        """
        Show statistics about loaded datasets
        """
        if not self.documents:
            print("\n⚠️  No datasets loaded. Please load datasets first (Option 1).")
            return
        
        print("\n" + "="*70)
        print(" " * 20 + "DATASET STATISTICS")
        print("="*70)
        
        total_docs = 0
        for theme in self.themes:
            if theme in self.documents:
                num_docs = len(self.documents[theme])
                total_docs += num_docs
                print(f"📄 {theme:<25} : {num_docs:>6} documents")
        
        print("="*70)
        print(f"📊 Total Documents: {total_docs}")
        print(f"📁 Total Datasets Loaded: {len(self.documents)}")
        print("="*70)
    
    def run(self):
        """
        Main program loop
        """
        print("\n🚀 Welcome to Information Retrieval System")
        print("📌 UTS Praktikum Penelusuran Informasi")
        
        while True:
            self.show_menu()
            choice = input("\n👉 Select option [1-5]: ").strip()
            
            if choice == '1':
                self.load_and_index_datasets()
                input("\nPress Enter to continue...")
                
            elif choice == '2':
                if not self.documents:
                    print("\n⚠️  Please load datasets first (Option 1).")
                    input("\nPress Enter to continue...")
                    continue
                
                query = input("\n🔍 Enter your search query: ").strip()
                if query:
                    top_n = input("📊 Number of results to display (default 5): ").strip()
                    top_n = int(top_n) if top_n.isdigit() else 5
                    
                    results = self.search_all_datasets(query, top_n)
                    self.display_results(results)
                else:
                    print("❌ Query cannot be empty!")
                
                input("\nPress Enter to continue...")
                
            elif choice == '3':
                if not self.documents:
                    print("\n⚠️  Please load datasets first (Option 1).")
                    input("\nPress Enter to continue...")
                    continue
                
                self.show_dataset_menu()
                dataset_choice = input(f"\n👉 Select dataset [1-{len(self.themes) + 1}]: ").strip()
                
                if dataset_choice.isdigit():
                    idx = int(dataset_choice) - 1
                    if idx == len(self.themes):
                        continue
                    elif 0 <= idx < len(self.themes):
                        theme = self.themes[idx]
                        query = input("\n🔍 Enter your search query: ").strip()
                        
                        if query:
                            top_n = input("📊 Number of results to display (default 5): ").strip()
                            top_n = int(top_n) if top_n.isdigit() else 5
                            
                            results = self.search_single_dataset(query, theme, top_n)
                            self.display_results(results)
                        else:
                            print("❌ Query cannot be empty!")
                    else:
                        print("❌ Invalid choice!")
                else:
                    print("❌ Invalid input!")
                
                input("\nPress Enter to continue...")
                
            elif choice == '4':
                self.show_statistics()
                input("\nPress Enter to continue...")
                
            elif choice == '5':
                print("\n👋 Thank you for using Information Retrieval System!")
                print("🎓 Goodbye!\n")
                break
                
            else:
                print("\n❌ Invalid option! Please select 1-5.")
                input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Create and run the IR system
    ir_system = InformationRetrievalSystem()
    ir_system.run()
