import 'package:flutter/material.dart';
import '../models/search_result.dart';
import '../services/api_service.dart';
import '../main.dart';

class SearchScreen extends StatefulWidget {
  final ApiService api;
  const SearchScreen({super.key, required this.api});

  @override
  State<SearchScreen> createState() => _SearchScreenState();
}

class _SearchScreenState extends State<SearchScreen> {
  final TextEditingController _queryCtl = TextEditingController();
  List<SearchResult> _results = [];
  bool _loading = false;
  bool _hasSearched = false;

  Future<void> _search() async {
    final q = _queryCtl.text.trim();
    if (q.isEmpty) return;
    setState(() { _loading = true; _hasSearched = true; });
    try {
      final results = await widget.api.searchAll(q);
      setState(() { _results = results; _loading = false; });
    } catch (e) {
      setState(() { _loading = false; });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Search error: $e')));
      }
    }
  }

  IconData _iconForType(String type) {
    switch (type) {
      case 'task': return Icons.checklist;
      case 'note': return Icons.description;
      case 'memory': return Icons.psychology;
      default: return Icons.article;
    }
  }

  @override
  void dispose() {
    _queryCtl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _queryCtl,
                    decoration: const InputDecoration(
                      hintText: 'Search tasks, notes, memories...',
                      prefixIcon: Icon(Icons.search),
                    ),
                    onSubmitted: (_) => _search(),
                  ),
                ),
                const SizedBox(width: 8),
                FilledButton(onPressed: _search, child: const Text('Search')),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Expanded(
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : !_hasSearched
                    ? Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.search, size: 48, color: context.jarvis.textSecondary),
                            const SizedBox(height: 12),
                            Text('Type a query and search.', style: TextStyle(color: context.jarvis.textSecondary)),
                          ],
                        ),
                      )
                    : _results.isEmpty
                        ? Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.search_off, size: 48, color: context.jarvis.textSecondary),
                                const SizedBox(height: 12),
                                Text('No results found.', style: TextStyle(color: context.jarvis.textSecondary)),
                              ],
                            ),
                          )
                        : ListView.builder(
                            padding: const EdgeInsets.symmetric(horizontal: 16),
                            itemCount: _results.length,
                            itemBuilder: (ctx, i) {
                              final r = _results[i];
                              return Card(
                                child: ListTile(
                                  leading: Icon(_iconForType(r.type)),
                                  title: Text(r.title),
                                  subtitle: Text('${r.type} · ${r.snippet}',
                                    maxLines: 2, overflow: TextOverflow.ellipsis,
                                    style: TextStyle(color: context.jarvis.textSecondary, fontSize: 12)),
                                ),
                              );
                            },
                          ),
          ),
        ],
      ),
    );
  }
}
