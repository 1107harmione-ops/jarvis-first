import 'package:flutter/material.dart';
import '../models/memory.dart';
import '../services/api_service.dart';
import '../main.dart';

class MemoryScreen extends StatefulWidget {
  final ApiService api;
  const MemoryScreen({super.key, required this.api});

  @override
  State<MemoryScreen> createState() => _MemoryScreenState();
}

class _MemoryScreenState extends State<MemoryScreen> {
  List<MemoryEntry> _entries = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() { _loading = true; _error = null; });
    try {
      final result = await widget.api.getMemories();
      setState(() { _entries = result.entries; _loading = false; });
    } catch (e) {
      setState(() { _error = e.toString(); _loading = false; });
    }
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
                  child: Text('Memory', style: Theme.of(context).textTheme.titleMedium),
                ),
                Text('${_entries.length} memories', style: TextStyle(color: context.jarvis.textSecondary, fontSize: 12)),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Text('Things Jarvis remembers about you', style: TextStyle(color: context.jarvis.textSecondary, fontSize: 12)),
          ),
          const SizedBox(height: 8),
          Expanded(
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : _error != null
                    ? Center(child: Text('Error: $_error'))
                    : _entries.isEmpty
                        ? Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.psychology, size: 48, color: context.jarvis.textSecondary),
                                const SizedBox(height: 12),
                                Text('No memories yet. Say "remember that..."', style: TextStyle(color: context.jarvis.textSecondary)),
                              ],
                            ),
                          )
                        : RefreshIndicator(
                            onRefresh: _load,
                            child: ListView.builder(
                              padding: const EdgeInsets.symmetric(horizontal: 16),
                              itemCount: _entries.length,
                              itemBuilder: (ctx, i) {
                                final m = _entries[i];
                                return Card(
                                  child: ListTile(
                                    leading: const Icon(Icons.psychology_outlined),
                                    title: Text(m.fact),
                                    subtitle: Row(
                                      children: [
                                        Text('${m.category} · importance ${m.importance.toStringAsFixed(1)}',
                                          style: TextStyle(color: context.jarvis.textSecondary, fontSize: 12)),
                                        if (m.createdAt != null) ...[
                                          const SizedBox(width: 8),
                                          Text('· ${m.createdAt!.month}/${m.createdAt!.day}/${m.createdAt!.year}',
                                            style: TextStyle(color: context.jarvis.textSecondary, fontSize: 12)),
                                        ],
                                      ],
                                    ),
                                    trailing: IconButton(
                                      icon: Icon(Icons.delete_outline, color: context.jarvis.red, size: 20),
                                      onPressed: () async {
                                        try {
                                          await widget.api.deleteMemory(m.id!);
                                          _load();
                                        } catch (e) {
                                          if (mounted) {
                                            ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
                                          }
                                        }
                                      },
                                    ),
                                  ),
                                );
                              },
                            ),
                          ),
          ),
        ],
      ),
    );
  }
}
