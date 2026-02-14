#!/usr/bin/env python3
"""
Knowledge-Graph Exporter - Enhanced Version

Extracts rich entity-relationship triples from multiple sources:
- MEMORY.md (structured bullet points)
- memory/*.md (daily session logs)
- TOOLS.md (tool definitions)
- skills/*/SKILL.md (skill definitions)
- TASKS.md (task definitions)

Produces triples with:
- subject, predicate, object
- subject_type, object_type (agent, skill, tool, service, file, user, cron, api, other)
- source_file
- confidence score
"""
import os
import re
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from datetime import datetime

WORKSPACE = Path("/root/.openclaw/workspace")
MEMORY_FILE = WORKSPACE / "MEMORY.md"
MEMORY_DIR = WORKSPACE / "memory"
TOOLS_FILE = WORKSPACE / "TOOLS.md"
SKILLS_DIR = WORKSPACE / "skills"
TASKS_FILE = WORKSPACE / "TASKS.md"
OUTPUT = Path("/root/.openclaw/workspace/projects/memory-engine/data/knowledge_graph.csv")

ACTION_VERBS = [
    'added', 'created', 'fixed', 'updated', 'implemented', 'removed',
    'patched', 'disabled', 'installed', 'wired', 'configured',
    'improved', 'refactored', 'enhanced', 'resolved', 'replaced',
    'removed', 'deleted', 'renamed', 'moved', 'started', 'stopped'
]

RELATION_PATTERNS = {
    'uses': [r'uses\s+(.+)', r'using\s+(.+)', r'with\s+(.+)'],
    'depends_on': [r'depends\s+on\s+(.+)', r'requires\s+(.+)', r'needs\s+(.+)'],
    'installed': [r'installed\s+(.+)', r'install\s+(.+)'],
    'runs_on': [r'runs\s+on\s+(.+)', r'running\s+on\s+(.+)'],
    'connected_to': [r'connected\s+to\s+(.+)', r'connects?\s+to\s+(.+)'],
    'configured_with': [r'configured\s+with\s+(.+)', r'config.*\s+(.+)'],
    'has_command': [r'command[:\s]+(.+)', r'commands?\s+(.+)'],
    'monitors': [r'monitors?\s+(.+)', r'monitoring\s+(.+)'],
    'supports': [r'supports?\s+(.+)', r'supports?\s+(.+)'],
    'created': [r'created\s+(.+)', r'creates?\s+(.+)'],
    'fixed': [r'fixed\s+(.+)', r'fixes?\s+(.+)'],
    'failed_with': [r'failed\s+(?:with|due to|error)\s+(.+)'],
}

ENTITY_PATTERNS = {
    'skill': [
        r'skills/([a-zA-Z0-9_-]+)/?',
        r'`?([a-z]+-[a-z]+)_skill`?',
        r'skill[:\s]+([a-zA-Z0-9_-]+)',
    ],
    'tool': [
        r'tools/([a-zA-Z0-9_]+\.py)',
        r'scripts/([a-zA-Z0-9_]+\.sh)',
        r'`([a-zA-Z0-9_]+\.py)`',
        r'`([a-zA-Z0-9_]+\.sh)`',
    ],
    'file': [
        r'`([^`]+\.(py|md|json|sh|txt|yml|yaml))`',
        r'([a-zA-Z0-9_/-]+\.(py|md|json|sh|txt|yml|yaml))',
    ],
    'service': [
        r'(postgres|postgresql)',
        r'(redis)',
        r'(nextcloud)',
        r'(ollama)',
        r'(openclaw[_-]?gateway)',
        r'(embedding[_-]?server)',
        r'(stt[_-]?server)',
        r'(tts[_-]?gen)?',
        r'(dashboard)',
    ],
    'api': [
        r'(OPENAI_API_KEY)',
        r'(GROQ_API_KEY)',
        r'(GEMINI_API_KEY)',
        r'(PERPLEXITY_API_KEY)',
        r'(OPENROUTER_API_KEY)',
    ],
    'port': [
        r'port[=:\s]+(\d{4,5})',
        r'(\d{4,5})[=:\s]+\w+',
    ],
    'cron': [
        r'cron[job]*[:\s]+([a-zA-Z0-9_-]+)',
        r'scheduled[:\s]+([a-zA-Z0-9_-]+)',
    ],
}

def normalize_entity(text: str) -> str:
    """Clean and normalize entity names."""
    text = text.strip()
    text = re.sub(r'^[-*]\s*', '', text)
    text = re.sub(r'`', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip('.,;:()[]{}')
    return text[:80]

def extract_entity(text: str, entity_type: str) -> List[str]:
    """Extract entities of a specific type from text."""
    entities = []
    patterns = ENTITY_PATTERNS.get(entity_type, [])
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities.extend([normalize_entity(m) for m in matches if m])
    return list(set(entities))

def detect_subject_type(text: str) -> str:
    """Detect the type of a subject."""
    text_lower = text.lower()
    
    if text_lower in ['agent', 'system', 'max', 'user', 'assistant', 'me']:
        return 'agent'
    
    if 'skill' in text_lower or '/' in text and text.startswith('skills'):
        return 'skill'
    
    if '.py' in text or '.sh' in text or 'tools/' in text or 'scripts/' in text:
        return 'tool'
    
    if any(svc in text_lower for svc in ['postgres', 'redis', 'nextcloud', 'ollama', 'gateway', 'stt', 'embedding', 'tts', 'dashboard']):
        return 'service'
    
    if 'api_key' in text_lower or '_api_key' in text_lower:
        return 'api'
    
    if 'cron' in text_lower or 'scheduled' in text_lower:
        return 'cron'
    
    if any(ext in text for ext in ['.py', '.md', '.json', '.sh', '.txt']):
        return 'file'
    
    return 'other'

def extract_from_memory_md(text: str, source: str) -> List[Dict]:
    """Extract triples from MEMORY.md format."""
    triples = []
    lines = text.splitlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        line = re.sub(r'^\d{4}-\d{2}-\d{2}:\s*', '', line)
        
        if line.startswith('- '):
            line = line[2:]
        
        date_match = re.match(r'^\d{4}-\d{2}-\d{2}:', line)
        if date_match:
            line = line[date_match.end():].strip()
        
        for verb in ACTION_VERBS:
            pattern = rf'^{verb}s?\s+(.+)$'
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                obj = normalize_entity(match.group(1))
                if obj and len(obj) > 1:
                    triples.append({
                        'subject': 'agent',
                        'predicate': verb,
                        'object': obj,
                        'subject_type': 'agent',
                        'object_type': detect_subject_type(obj),
                        'source_file': source,
                        'confidence': 0.9
                    })
                    
                    nested = re.findall(r'[-*]\s+([^-].+)', line)
                    for n in nested[:5]:
                        n = normalize_entity(n)
                        if n and len(n) > 2:
                            triples.append({
                                'subject': obj,
                                'predicate': 'has_component',
                                'object': n,
                                'subject_type': detect_subject_type(obj),
                                'object_type': detect_subject_type(n),
                                'source_file': source,
                                'confidence': 0.7
                            })
                break
        
        if ' uses ' in line.lower():
            parts = re.split(r'\s+uses\s+', line, flags=re.IGNORECASE)
            if len(parts) == 2:
                subj = normalize_entity(parts[0].split()[-1])
                obj = normalize_entity(parts[1].split(',')[0])
                if subj and obj:
                    triples.append({
                        'subject': subj,
                        'predicate': 'uses',
                        'object': obj,
                        'subject_type': detect_subject_type(subj),
                        'object_type': detect_subject_type(obj),
                        'source_file': source,
                        'confidence': 0.8
                    })
        
        for pred, patterns in RELATION_PATTERNS.items():
            if pred in ['created', 'fixed', 'uses']:
                continue
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    subj = line.split(match.group(0).split()[0])[0].strip()
                    subj = normalize_entity(subj.split(',')[0])
                    obj = normalize_entity(match.group(1).split(',')[0])
                    if subj and obj and len(subj) > 1 and len(obj) > 1:
                        triples.append({
                            'subject': subj,
                            'predicate': pred,
                            'object': obj,
                            'subject_type': detect_subject_type(subj),
                            'object_type': detect_subject_type(obj),
                            'source_file': source,
                            'confidence': 0.8
                        })
                    break
    
    return triples

def extract_from_tools_md(text: str, source: str) -> List[Dict]:
    """Extract triples from TOOLS.md format."""
    triples = []
    lines = text.splitlines()
    
    in_tool_section = False
    current_tool = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('## '):
            in_tool_section = 'Tool' in line or 'Script' in line
            tool_match = re.search(r'##\s+(?:Tool|Script)[:\s]+([^\(]+)', line)
            if tool_match:
                current_tool = normalize_entity(tool_match.group(1).strip())
        
        if not current_tool:
            continue
        
        file_refs = extract_entity(line, 'file')
        for f in file_refs[:3]:
            triples.append({
                'subject': current_tool,
                'predicate': 'implemented_in',
                'object': f,
                'subject_type': 'tool',
                'object_type': 'file',
                'source_file': source,
                'confidence': 0.9
            })
        
        service_refs = extract_entity(line, 'service')
        for s in service_refs[:2]:
            triples.append({
                'subject': current_tool,
                'predicate': 'uses_service',
                'object': s,
                'subject_type': 'tool',
                'object_type': 'service',
                'source_file': source,
                'confidence': 0.8
            })
        
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3 and parts[1]:
                cmd = normalize_entity(parts[1])
                triples.append({
                    'subject': current_tool,
                    'predicate': 'has_command',
                    'object': cmd,
                    'subject_type': 'tool',
                    'object_type': 'other',
                    'source_file': source,
                    'confidence': 0.8
                })
    
    return triples

def extract_from_skills_dir(skills_dir: Path, source_prefix: str) -> List[Dict]:
    """Extract triples from skills directory."""
    triples = []
    
    if not skills_dir.exists():
        return triples
    
    for skill_path in skills_dir.iterdir():
        if not skill_path.is_dir():
            continue
        
        skill_name = skill_path.name
        
        skill_md = skill_path / 'SKILL.md'
        meta_json = skill_path / '_meta.json'
        
        triples.append({
            'subject': 'agent',
            'predicate': 'has_skill',
            'object': skill_name,
            'subject_type': 'agent',
            'object_type': 'skill',
            'source_file': f"{source_prefix}/{skill_name}",
            'confidence': 1.0
        })
        
        if skill_md.exists():
            content = skill_md.read_text(encoding='utf-8')
            
            commands = re.findall(r'[`]?(\w+)[`]?\s*[-–]\s*(.+)', content)
            for cmd, desc in commands[:5]:
                triples.append({
                    'subject': skill_name,
                    'predicate': 'has_command',
                    'object': f"{cmd}: {desc[:40]}",
                    'subject_type': 'skill',
                    'object_type': 'other',
                    'source_file': f"{source_prefix}/{skill_name}/SKILL.md",
                    'confidence': 0.9
                })
            
            for svc in extract_entity(content, 'service'):
                triples.append({
                    'subject': skill_name,
                    'predicate': 'uses_service',
                    'object': svc,
                    'subject_type': 'skill',
                    'object_type': 'service',
                    'source_file': f"{source_prefix}/{skill_name}/SKILL.md",
                    'confidence': 0.8
                })
        
        if meta_json.exists():
            try:
                meta = json.loads(meta_json.read_text(encoding='utf-8'))
                if 'commands' in meta:
                    for cmd in meta['commands'][:5]:
                        triples.append({
                            'subject': skill_name,
                            'predicate': 'has_command',
                            'object': cmd,
                            'subject_type': 'skill',
                            'object_type': 'other',
                            'source_file': f"{source_prefix}/{skill_name}/_meta.json",
                            'confidence': 0.9
                        })
            except:
                pass
    
    return triples

def extract_from_tasks_md(text: str, source: str) -> List[Dict]:
    """Extract triples from TASKS.md."""
    triples = []
    
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        
        if line.startswith('- ['):
            status = 'pending' in line.lower() if 'pending' in line.lower() else 'done'
            task_match = re.search(r'- \[.?\]\s+(.+)', line)
            if task_match:
                task = normalize_entity(task_match.group(1))
                triples.append({
                    'subject': 'agent',
                    'predicate': 'has_task',
                    'object': task,
                    'subject_type': 'agent',
                    'object_type': 'other',
                    'source_file': source,
                    'confidence': 0.7
                })
    
    return triples

def extract_from_daily_log(text: str, source: str) -> List[Dict]:
    """Extract triples from daily memory logs."""
    triples = []
    lines = text.splitlines()
    
    in_summary = False
    for line in lines:
        line = line.strip()
        
        if 'conversation summary' in line.lower():
            in_summary = True
            continue
        
        if line.startswith('## '):
            in_summary = False
            continue
        
        if in_summary and line:
            for verb in ACTION_VERBS[:8]:
                pattern = rf'^{verb}s?\s+(.+)$'
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    obj = normalize_entity(match.group(1))
                    if obj and len(obj) > 2:
                        triples.append({
                            'subject': 'agent',
                            'predicate': verb,
                            'object': obj,
                            'subject_type': 'agent',
                            'object_type': detect_subject_type(obj),
                            'source_file': source,
                            'confidence': 0.6
                        })
                    break
    
    return triples

def gather_sources() -> List:
    """Gather all source files and their content."""
    sources = []
    
    if MEMORY_FILE.exists():
        sources.append(('memory', MEMORY_FILE.read_text(encoding='utf-8'), "MEMORY.md"))
    
    if MEMORY_DIR.exists():
        for fname in sorted(MEMORY_DIR.glob("*.md")):
            if 'memory-engine' not in fname.name:
                sources.append(('daily', fname.read_text(encoding='utf-8'), f"memory/{fname.name}"))
    
    if TOOLS_FILE.exists():
        sources.append(('tools', TOOLS_FILE.read_text(encoding='utf-8'), "TOOLS.md"))
    
    if TASKS_FILE.exists():
        sources.append(('tasks', TASKS_FILE.read_text(encoding='utf-8'), "TASKS.md"))
    
    skill_triples = extract_from_skills_dir(SKILLS_DIR, "skills")
    if skill_triples:
        sources.append(('skills', skill_triples, "skills/"))
    
    return sources

def deduplicate(triples: List[Dict]) -> List[Dict]:
    """Remove duplicate triples."""
    seen = set()
    unique = []
    
    for t in triples:
        key = (t['subject'].lower(), t['predicate'].lower(), t['object'].lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)
    
    return unique

def build_graph():
    """Main function to build the knowledge graph."""
    print("Starting knowledge graph export...")
    
    all_triples = []
    
    sources = gather_sources()
    print(f"Processing {len(sources)} sources...")
    
    for item in sources:
        source_type = item[0]
        content = item[1]
        src = item[2]
        
        if source_type == 'skills':
            all_triples.extend(content)
        elif source_type == 'daily':
            all_triples.extend(extract_from_daily_log(content, src))
        elif source_type == 'tools':
            all_triples.extend(extract_from_tools_md(content, src))
        elif source_type == 'tasks':
            all_triples.extend(extract_from_tasks_md(content, src))
        elif source_type == 'memory':
            all_triples.extend(extract_from_memory_md(content, src))
    
    print(f"Extracted {len(all_triples)} raw triples")
    
    all_triples = deduplicate(all_triples)
    print(f"After deduplication: {len(all_triples)} triples")
    
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'subject', 'predicate', 'object', 'subject_type', 'object_type', 'source_file', 'confidence'
        ])
        writer.writeheader()
        writer.writerows(all_triples)
    
    subject_types = {}
    object_types = {}
    predicates = {}
    
    for t in all_triples:
        subject_types[t['subject_type']] = subject_types.get(t['subject_type'], 0) + 1
        object_types[t['object_type']] = object_types.get(t['object_type'], 0) + 1
        predicates[t['predicate']] = predicates.get(t['predicate'], 0) + 1
    
    print(f"\nKnowledge graph written to {OUTPUT}")
    print(f"Total triples: {len(all_triples)}")
    print(f"\nSubject types: {dict(sorted(subject_types.items()))}")
    print(f"Object types: {dict(sorted(object_types.items()))}")
    print(f"Top predicates: {dict(sorted(predicates.items(), key=lambda x: -x[1])[:10])}")
    
    return all_triples

if __name__ == "__main__":
    build_graph()
