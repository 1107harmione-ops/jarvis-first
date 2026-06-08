import 'package:flutter/material.dart';
import '../models/note.dart';
import '../services/api_service.dart';
import '../main.dart';

class NotesScreen extends StatefulWidget {
  final ApiService api;
  const NotesScreen({super.key, required this.api});

  @override
  State<NotesScreen> createState() => _NotesScreenState();
}

class _NotesScreenState extends State<NotesScreen> {
  List<Note> _notes = [];
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
      final result = await widget.api.getNotes();
      setState(() { _notes = result.notes; _loading = false; });
    } catch (e) {
      setState(() { _error = e.toString(); _loading = false; });
    }
  }

  Future<void> _createOrUpdate({Note? note}) async {
    final isEdit = note != null;
    final titleCtl = TextEditingController(text: note?.title ?? '');
    final contentCtl = TextEditingController(text: note?.content ?? '');

    final result = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(isEdit ? 'Edit Note' : 'New Note'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(controller: titleCtl, decoration: const InputDecoration(labelText: 'Title'), autofocus: true),
            const SizedBox(height: 12),
            TextField(controller: contentCtl, decoration: const InputDecoration(labelText: 'Content'), maxLines: 5),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: Text(isEdit ? 'Update' : 'Create')),
        ],
      ),
    );

    if (result == true && titleCtl.text.trim().isNotEmpty) {
      try {
        if (isEdit) {
          await widget.api.updateNote(note!.id!, title: titleCtl.text.trim(), content: contentCtl.text.trim());
        } else {
          await widget.api.createNote(title: titleCtl.text.trim(), content: contentCtl.text.trim());
        }
        _load();
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
        }
      }
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
                Text('Notes', style: Theme.of(context).textTheme.titleMedium),
                const Spacer(),
                FilledButton.icon(
                  icon: const Icon(Icons.add, size: 18),
                  label: const Text('New Note'),
                  onPressed: () => _createOrUpdate(),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Expanded(
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : _error != null
                    ? Center(child: Text('Error: $_error'))
                    : _notes.isEmpty
                        ? Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.note, size: 48, color: context.jarvis.textSecondary),
                                const SizedBox(height: 12),
                                Text('No notes yet.', style: TextStyle(color: context.jarvis.textSecondary)),
                              ],
                            ),
                          )
                        : RefreshIndicator(
                            onRefresh: _load,
                            child: ListView.builder(
                              padding: const EdgeInsets.symmetric(horizontal: 16),
                              itemCount: _notes.length,
                              itemBuilder: (ctx, i) {
                                final n = _notes[i];
                                return Card(
                                  child: ListTile(
                                    leading: const Icon(Icons.description_outlined),
                                    title: Text(n.title),
                                    subtitle: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        if (n.content != null && n.content!.isNotEmpty)
                                          Text(n.content!, maxLines: 2, overflow: TextOverflow.ellipsis,
                                            style: TextStyle(color: context.jarvis.textSecondary, fontSize: 12)),
                                        if (n.tags != null && n.tags!.isNotEmpty)
                                          Text(n.tags!, style: const TextStyle(fontSize: 11, color: Colors.grey)),
                                      ],
                                    ),
                                    trailing: PopupMenuButton(
                                      itemBuilder: (_) => [
                                        PopupMenuItem(child: const Text('Edit'), onTap: () => _createOrUpdate(note: n)),
                                        PopupMenuItem(child: Text('Delete', style: TextStyle(color: context.jarvis.red)), onTap: () async {
                                          try {
                                            await widget.api.deleteNote(n.id!);
                                            _load();
                                          } catch (e) {
                                            if (mounted) {
                                              ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
                                            }
                                          }
                                        }),
                                      ],
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
