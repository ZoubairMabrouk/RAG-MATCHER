// import React, { useState } from 'react';
// import { Database, FileJson, GitCompare, CheckCircle, AlertCircle, FileCode, Layers } from 'lucide-react';

// const DBEvolutionDemo = () => {
//   const [activeTab, setActiveTab] = useState('overview');
//   const [selectedFile, setSelectedFile] = useState(null);
  
//   const sampleUSchema = {
//     entities: [
//       {
//         name: "Customer",
//         attributes: [
//           { name: "id", type: "string", required: true },
//           { name: "name", type: "string", required: true },
//           { name: "email", type: "string" },
//           { name: "created_at", type: "timestamp" }
//         ]
//       },
//       {
//         name: "Order",
//         attributes: [
//           { name: "id", type: "string", required: true },
//           { name: "customer_id", type: "string", required: true },
//           { name: "total", type: "decimal", required: true },
//           { name: "status", type: "string" }
//         ],
//         relationships: [
//           { type: "belongsTo", entity: "Customer", key: "customer_id" }
//         ]
//       }
//     ]
//   };

//   const currentSchema = {
//     tables: [
//       {
//         name: "customers",
//         columns: [
//           { name: "id", type: "uuid", nullable: false, pk: true },
//           { name: "name", type: "varchar(255)", nullable: false },
//           { name: "created_at", type: "timestamp", nullable: true }
//         ]
//       }
//     ]
//   };

//   const evolutionPlan = {
//     changes: [
//       {
//         type: "ADD_COLUMN",
//         table: "customers",
//         column: "email",
//         definition: "VARCHAR(255)",
//         reason: "U-Schema requires email attribute not present in current schema"
//       },
//       {
//         type: "CREATE_TABLE",
//         table: "orders",
//         columns: [
//           { name: "id", type: "UUID PRIMARY KEY" },
//           { name: "customer_id", type: "UUID NOT NULL" },
//           { name: "total", type: "DECIMAL(10,2) NOT NULL" },
//           { name: "status", type: "VARCHAR(50)" }
//         ],
//         reason: "New entity 'Order' requires corresponding table"
//       },
//       {
//         type: "ADD_FOREIGN_KEY",
//         table: "orders",
//         column: "customer_id",
//         references: "customers(id)",
//         reason: "Relationship defined in U-Schema"
//       }
//     ],
//     sql: [
//       "ALTER TABLE customers ADD COLUMN email VARCHAR(255);",
//       "CREATE TABLE orders (\n  id UUID PRIMARY KEY,\n  customer_id UUID NOT NULL,\n  total DECIMAL(10,2) NOT NULL,\n  status VARCHAR(50)\n);",
//       "ALTER TABLE orders ADD CONSTRAINT fk_orders_customer FOREIGN KEY (customer_id) REFERENCES customers(id);"
//     ]
//   };

//   const architectureLayers = [
//     {
//       name: "Presentation Layer",
//       components: ["CLI Interface", "REST API", "Web Dashboard"],
//       color: "bg-blue-100 border-blue-300"
//     },
//     {
//       name: "Application Layer",
//       components: ["Evolution Orchestrator", "Validation Pipeline", "Report Generator"],
//       color: "bg-green-100 border-green-300"
//     },
//     {
//       name: "Domain Layer",
//       components: ["Schema Parser", "Diff Engine", "Rule Engine", "Migration Builder"],
//       color: "bg-purple-100 border-purple-300"
//     },
//     {
//       name: "Infrastructure Layer",
//       components: ["RAG System", "Vector Store", "LLM Client", "Database Inspector"],
//       color: "bg-orange-100 border-orange-300"
//     }
//   ];

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
//       <div className="max-w-7xl mx-auto">
//         {/* Header */}
//         <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
//           <div className="flex items-center gap-3 mb-4">
//             <Database className="w-10 h-10 text-blue-600" />
//             <div>
//               <h1 className="text-3xl font-bold text-gray-900">Database Evolution System</h1>
//               <p className="text-gray-600">LLM-Powered Schema Migration with RAG</p>
//             </div>
//           </div>
          
//           <div className="flex gap-2 mt-4">
//             <button
//               onClick={() => setActiveTab('overview')}
//               className={`px-4 py-2 rounded-lg font-medium transition ${
//                 activeTab === 'overview'
//                   ? 'bg-blue-600 text-white'
//                   : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
//               }`}
//             >
//               Overview
//             </button>
//             <button
//               onClick={() => setActiveTab('architecture')}
//               className={`px-4 py-2 rounded-lg font-medium transition ${
//                 activeTab === 'architecture'
//                   ? 'bg-blue-600 text-white'
//                   : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
//               }`}
//             >
//               Architecture
//             </button>
//             <button
//               onClick={() => setActiveTab('demo')}
//               className={`px-4 py-2 rounded-lg font-medium transition ${
//                 activeTab === 'demo'
//                   ? 'bg-blue-600 text-white'
//                   : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
//               }`}
//             >
//               Live Demo
//             </button>
//             <button
//               onClick={() => setActiveTab('code')}
//               className={`px-4 py-2 rounded-lg font-medium transition ${
//                 activeTab === 'code'
//                   ? 'bg-blue-600 text-white'
//                   : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
//               }`}
//             >
//               Code Structure
//             </button>
//           </div>
//         </div>

//         {/* Overview Tab */}
//         {activeTab === 'overview' && (
//           <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
//             <div className="bg-white rounded-lg shadow-lg p-6">
//               <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
//                 <FileJson className="w-6 h-6 text-blue-600" />
//                 System Capabilities
//               </h2>
//               <ul className="space-y-3">
//                 <li className="flex items-start gap-2">
//                   <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
//                   <span className="text-gray-700">Ingests U-Schema JSON (NoSQL-oriented conceptual model)</span>
//                 </li>
//                 <li className="flex items-start gap-2">
//                   <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
//                   <span className="text-gray-700">Introspects current relational database schema</span>
//                 </li>
//                 <li className="flex items-start gap-2">
//                   <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
//                   <span className="text-gray-700">Uses RAG to retrieve relevant schema context</span>
//                 </li>
//                 <li className="flex items-start gap-2">
//                   <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
//                   <span className="text-gray-700">Generates human-readable evolution descriptions</span>
//                 </li>
//                 <li className="flex items-start gap-2">
//                   <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
//                   <span className="text-gray-700">Produces executable DDL/migration scripts</span>
//                 </li>
//                 <li className="flex items-start gap-2">
//                   <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
//                   <span className="text-gray-700">Validates migrations before execution</span>
//                 </li>
//               </ul>
//             </div>

//             <div className="bg-white rounded-lg shadow-lg p-6">
//               <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
//                 <GitCompare className="w-6 h-6 text-purple-600" />
//                 Key Features
//               </h2>
//               <div className="space-y-4">
//                 <div className="border-l-4 border-blue-500 pl-4">
//                   <h3 className="font-semibold text-gray-900">RAG-Powered Context</h3>
//                   <p className="text-sm text-gray-600">Vector embeddings of schema, rules, and conventions</p>
//                 </div>
//                 <div className="border-l-4 border-green-500 pl-4">
//                   <h3 className="font-semibold text-gray-900">SOLID Design</h3>
//                   <p className="text-sm text-gray-600">Clean architecture with clear separation of concerns</p>
//                 </div>
//                 <div className="border-l-4 border-purple-500 pl-4">
//                   <h3 className="font-semibold text-gray-900">Multi-Model Support</h3>
//                   <p className="text-sm text-gray-600">Works with various LLMs (OpenAI, Anthropic, local)</p>
//                 </div>
//                 <div className="border-l-4 border-orange-500 pl-4">
//                   <h3 className="font-semibold text-gray-900">Safety First</h3>
//                   <p className="text-sm text-gray-600">Validation, dry-runs, and rollback support</p>
//                 </div>
//               </div>
//             </div>
//           </div>
//         )}

//         {/* Architecture Tab */}
//         {activeTab === 'architecture' && (
//           <div className="space-y-6">
//             <div className="bg-white rounded-lg shadow-lg p-6">
//               <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
//                 <Layers className="w-6 h-6 text-blue-600" />
//                 System Architecture (Layered)
//               </h2>
//               <div className="space-y-4">
//                 {architectureLayers.map((layer, idx) => (
//                   <div key={idx} className={`border-2 rounded-lg p-4 ${layer.color}`}>
//                     <h3 className="font-bold text-gray-900 mb-2">{layer.name}</h3>
//                     <div className="flex flex-wrap gap-2">
//                       {layer.components.map((comp, i) => (
//                         <span key={i} className="bg-white px-3 py-1 rounded-full text-sm font-medium text-gray-700 shadow-sm">
//                           {comp}
//                         </span>
//                       ))}
//                     </div>
//                   </div>
//                 ))}
//               </div>
//             </div>

//             <div className="bg-white rounded-lg shadow-lg p-6">
//               <h2 className="text-xl font-bold text-gray-900 mb-4">Data Flow</h2>
//               <div className="space-y-3 text-sm">
//                 <div className="flex items-center gap-3">
//                   <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold">1</div>
//                   <div className="flex-1 bg-blue-50 p-3 rounded">
//                     <strong>Input:</strong> U-Schema JSON + Design Rules
//                   </div>
//                 </div>
//                 <div className="flex items-center gap-3">
//                   <div className="w-8 h-8 rounded-full bg-green-600 text-white flex items-center justify-center font-bold">2</div>
//                   <div className="flex-1 bg-green-50 p-3 rounded">
//                     <strong>Introspection:</strong> Extract current DB schema, metadata, statistics
//                   </div>
//                 </div>
//                 <div className="flex items-center gap-3">
//                   <div className="w-8 h-8 rounded-full bg-purple-600 text-white flex items-center justify-center font-bold">3</div>
//                   <div className="flex-1 bg-purple-50 p-3 rounded">
//                     <strong>RAG Retrieval:</strong> Find relevant tables, columns, rules via vector search
//                   </div>
//                 </div>
//                 <div className="flex items-center gap-3">
//                   <div className="w-8 h-8 rounded-full bg-orange-600 text-white flex items-center justify-center font-bold">4</div>
//                   <div className="flex-1 bg-orange-50 p-3 rounded">
//                     <strong>Diff Analysis:</strong> Symbolic comparison + LLM-powered semantic analysis
//                   </div>
//                 </div>
//                 <div className="flex items-center gap-3">
//                   <div className="w-8 h-8 rounded-full bg-red-600 text-white flex items-center justify-center font-bold">5</div>
//                   <div className="flex-1 bg-red-50 p-3 rounded">
//                     <strong>Evolution Plan:</strong> Generate structured plan + SQL + explanations
//                   </div>
//                 </div>
//                 <div className="flex items-center gap-3">
//                   <div className="w-8 h-8 rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold">6</div>
//                   <div className="flex-1 bg-indigo-50 p-3 rounded">
//                     <strong>Validation:</strong> Parse SQL, dry-run, check invariants
//                   </div>
//                 </div>
//               </div>
//             </div>
//           </div>
//         )}

//         {/* Demo Tab */}
//         {activeTab === 'demo' && (
//           <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
//             <div className="space-y-6">
//               <div className="bg-white rounded-lg shadow-lg p-6">
//                 <h2 className="text-lg font-bold text-gray-900 mb-3 flex items-center gap-2">
//                   <FileJson className="w-5 h-5 text-blue-600" />
//                   U-Schema (Target)
//                 </h2>
//                 <pre className="bg-gray-50 p-4 rounded text-xs overflow-auto max-h-64">
//                   {JSON.stringify(sampleUSchema, null, 2)}
//                 </pre>
//               </div>

//               <div className="bg-white rounded-lg shadow-lg p-6">
//                 <h2 className="text-lg font-bold text-gray-900 mb-3 flex items-center gap-2">
//                   <Database className="w-5 h-5 text-green-600" />
//                   Current Schema
//                 </h2>
//                 <pre className="bg-gray-50 p-4 rounded text-xs overflow-auto max-h-64">
//                   {JSON.stringify(currentSchema, null, 2)}
//                 </pre>
//               </div>
//             </div>

//             <div className="space-y-6">
//               <div className="bg-white rounded-lg shadow-lg p-6">
//                 <h2 className="text-lg font-bold text-gray-900 mb-3 flex items-center gap-2">
//                   <GitCompare className="w-5 h-5 text-purple-600" />
//                   Evolution Plan
//                 </h2>
//                 <div className="space-y-3">
//                   {evolutionPlan.changes.map((change, idx) => (
//                     <div key={idx} className="border-l-4 border-purple-500 bg-purple-50 p-3 rounded">
//                       <div className="font-semibold text-sm text-purple-900">{change.type}</div>
//                       <div className="text-xs text-gray-700 mt-1">
//                         {change.table && <span className="font-mono bg-white px-2 py-0.5 rounded">{change.table}</span>}
//                         {change.column && <span className="ml-2 font-mono bg-white px-2 py-0.5 rounded">{change.column}</span>}
//                       </div>
//                       <div className="text-xs text-gray-600 mt-2">{change.reason}</div>
//                     </div>
//                   ))}
//                 </div>
//               </div>

//               <div className="bg-white rounded-lg shadow-lg p-6">
//                 <h2 className="text-lg font-bold text-gray-900 mb-3 flex items-center gap-2">
//                   <FileCode className="w-5 h-5 text-orange-600" />
//                   Generated SQL
//                 </h2>
//                 <pre className="bg-gray-900 text-green-400 p-4 rounded text-xs overflow-auto max-h-64 font-mono">
//                   {evolutionPlan.sql.join('\n\n')}
//                 </pre>
//               </div>
//             </div>
//           </div>
//         )}

//         {/* Code Structure Tab */}
//         {activeTab === 'code' && (
//           <div className="bg-white rounded-lg shadow-lg p-6">
//             <h2 className="text-xl font-bold text-gray-900 mb-4">Project Structure</h2>
//             <div className="font-mono text-sm space-y-1 text-gray-700">
//               <div>📁 db-evolution-system/</div>
//               <div className="ml-4">📁 src/</div>
//               <div className="ml-8">📁 domain/</div>
//               <div className="ml-12 text-blue-600">├── entities/ (USchema, Table, Column, Rule)</div>
//               <div className="ml-12 text-blue-600">├── repositories/ (ISchemaRepository, IRuleRepository)</div>
//               <div className="ml-12 text-blue-600">└── services/ (DiffEngine, RuleEngine, MigrationBuilder)</div>
//               <div className="ml-8">📁 application/</div>
//               <div className="ml-12 text-green-600">├── use_cases/ (AnalyzeEvolution, GenerateMigration)</div>
//               <div className="ml-12 text-green-600">├── dtos/ (EvolutionRequest, EvolutionResponse)</div>
//               <div className="ml-12 text-green-600">└── orchestrators/ (EvolutionOrchestrator)</div>
//               <div className="ml-8">📁 infrastructure/</div>
//               <div className="ml-12 text-purple-600">├── rag/ (VectorStore, EmbeddingService, Retriever)</div>
//               <div className="ml-12 text-purple-600">├── llm/ (LLMClient, OpenAIProvider, AnthropicProvider)</div>
//               <div className="ml-12 text-purple-600">├── database/ (PostgresInspector, MySQLInspector)</div>
//               <div className="ml-12 text-purple-600">└── validators/ (SQLValidator, SafetyValidator)</div>
//               <div className="ml-8">📁 presentation/</div>
//               <div className="ml-12 text-orange-600">├── cli/ (CLI commands)</div>
//               <div className="ml-12 text-orange-600">├── api/ (FastAPI/Flask endpoints)</div>
//               <div className="ml-12 text-orange-600">└── web/ (Dashboard components)</div>
//               <div className="ml-4">📁 tests/</div>
//               <div className="ml-8">├── unit/</div>
//               <div className="ml-8">├── integration/</div>
//               <div className="ml-8">└── e2e/</div>
//               <div className="ml-4">📄 pyproject.toml</div>
//               <div className="ml-4">📄 README.md</div>
//             </div>
//           </div>
//         )}

//         {/* Footer */}
//         <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg shadow-lg p-6 mt-6 text-white">
//           <h3 className="text-lg font-bold mb-2">🚀 Ready to Build</h3>
//           <p className="text-blue-100 mb-4">
//             This system follows SOLID principles with clear separation of concerns, making it maintainable, testable, and extensible.
//           </p>
//           <div className="flex gap-2 flex-wrap">
//             <span className="bg-white bg-opacity-20 px-3 py-1 rounded-full text-sm">Python 3.11+</span>
//             <span className="bg-white bg-opacity-20 px-3 py-1 rounded-full text-sm">PostgreSQL/MySQL</span>
//             <span className="bg-white bg-opacity-20 px-3 py-1 rounded-full text-sm">FAISS/pgvector</span>
//             <span className="bg-white bg-opacity-20 px-3 py-1 rounded-full text-sm">OpenAI/Anthropic</span>
//             <span className="bg-white bg-opacity-20 px-3 py-1 rounded-full text-sm">FastAPI</span>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default DBEvolutionDemo;