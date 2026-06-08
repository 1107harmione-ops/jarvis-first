import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import '../models/task.dart';
import '../models/note.dart';
import '../models/reminder.dart';
import '../models/memory.dart';
import '../models/search_result.dart';

class ApiService extends ChangeNotifier {
  // Change this to your backend URL
  static const String _baseUrl = 'http://10.0.2.2:8000'; // Android emulator -> host
  // For iOS simulator, use: 'http://localhost:8000'
  // For physical device, use your computer's local IP

  final http.Client _client;

  ApiService({http.Client? client}) : _client = client ?? http.Client();

  Map<String, String> get _headers => {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      };

  Future<dynamic> _request(
    String method,
    String path, {
    Map<String, String>? queryParams,
    Object? body,
  }) async {
    final uri = Uri.parse('$_baseUrl$path').replace(queryParameters: queryParams);
    final msg = '$method $uri';

    http.Response response;
    try {
      response = await _client.send(
        http.Request(method, uri)
          ..headers.addAll(_headers)
          ..body = body != null ? jsonEncode(body) : '',
      ).then((r) => http.Response.fromStream(r));
    } catch (e) {
      debugPrint('$msg FAILED: $e');
      rethrow;
    }

    if (response.statusCode == 204) return null;

    final decoded = jsonDecode(response.body);
    if (response.statusCode >= 400) {
      final err = decoded is Map ? (decoded['detail'] ?? decoded['error'] ?? response.body) : response.body;
      throw ApiException(err.toString(), response.statusCode);
    }
    return decoded;
  }

  // ── Tasks ──

  Future<TaskListResponse> getTasks({String? status, String? priority}) async {
    final params = <String, String>{};
    if (status != null && status.isNotEmpty) params['status'] = status;
    if (priority != null && priority.isNotEmpty) params['priority'] = priority;
    final data = await _request('GET', '/api/tasks', queryParams: params.isEmpty ? null : params);
    return TaskListResponse.fromJson(data);
  }

  Future<Task> createTask({required String title, String? description, String? priority}) async {
    final data = await _request('POST', '/api/tasks', body: {
      'title': title,
      if (description != null) 'description': description,
      'priority': priority ?? 'medium',
    });
    return Task.fromJson(data);
  }

  Future<Task> getTask(int id) async {
    final data = await _request('GET', '/api/tasks/$id');
    return Task.fromJson(data);
  }

  Future<Task> updateTask(int id, {String? title, String? description, String? priority, String? status}) async {
    final body = <String, dynamic>{};
    if (title != null) body['title'] = title;
    if (description != null) body['description'] = description;
    if (priority != null) body['priority'] = priority;
    if (status != null) body['status'] = status;
    final data = await _request('PATCH', '/api/tasks/$id', body: body);
    return Task.fromJson(data);
  }

  Future<Task> completeTask(int id) async {
    final data = await _request('PATCH', '/api/tasks/$id/complete');
    return Task.fromJson(data);
  }

  Future<void> deleteTask(int id) async {
    await _request('DELETE', '/api/tasks/$id');
  }

  Future<List<Task>> searchTasks(String query) async {
    final data = await _request('GET', '/api/tasks/search', queryParams: {'q': query});
    return (data['tasks'] as List?)?.map((e) => Task.fromJson(e)).toList() ?? [];
  }

  // ── Notes ──

  Future<NoteListResponse> getNotes({int limit = 50}) async {
    final data = await _request('GET', '/api/notes', queryParams: {'limit': '$limit'});
    return NoteListResponse.fromJson(data);
  }

  Future<Note> createNote({required String title, String? content, String? tags}) async {
    final data = await _request('POST', '/api/notes', body: {
      'title': title,
      if (content != null) 'content': content,
      if (tags != null) 'tags': tags,
    });
    return Note.fromJson(data);
  }

  Future<Note> getNote(int id) async {
    final data = await _request('GET', '/api/notes/$id');
    return Note.fromJson(data);
  }

  Future<Note> updateNote(int id, {String? title, String? content, String? tags}) async {
    final body = <String, dynamic>{};
    if (title != null) body['title'] = title;
    if (content != null) body['content'] = content;
    if (tags != null) body['tags'] = tags;
    final data = await _request('PATCH', '/api/notes/$id', body: body);
    return Note.fromJson(data);
  }

  Future<void> deleteNote(int id) async {
    await _request('DELETE', '/api/notes/$id');
  }

  // ── Reminders ──

  Future<ReminderListResponse> getReminders() async {
    final data = await _request('GET', '/api/reminders');
    return ReminderListResponse.fromJson(data);
  }

  Future<Reminder> createReminder({required String title, required String reminderTime}) async {
    final data = await _request('POST', '/api/reminders', body: {
      'title': title,
      'reminder_time': reminderTime,
    });
    return Reminder.fromJson(data);
  }

  Future<void> deleteReminder(int id) async {
    await _request('DELETE', '/api/reminders/$id');
  }

  // ── Memory ──

  Future<MemoryListResponse> getMemories() async {
    final data = await _request('GET', '/api/memory');
    return MemoryListResponse.fromJson(data);
  }

  Future<MemoryEntry> storeMemory({required String fact, String? category, double importance = 0.5}) async {
    final data = await _request('POST', '/api/memory', body: {
      'fact': fact,
      if (category != null) 'category': category,
      'importance': importance,
    });
    return MemoryEntry.fromJson(data);
  }

  Future<void> deleteMemory(int id) async {
    await _request('DELETE', '/api/memory/$id');
  }

  // ── Voice ──

  Future<Map<String, dynamic>> sendVoiceCommand(String text) async {
    final data = await _request('POST', '/api/voice/command', body: {'text': text});
    return Map<String, dynamic>.from(data);
  }

  // ── Search ──

  Future<List<SearchResult>> searchAll(String query, {String? type}) async {
    final params = <String, String>{'q': query};
    if (type != null && type.isNotEmpty) params['type'] = type;
    final data = await _request('GET', '/api/search', queryParams: params);
    final results = data['results'] as List?;
    return results?.map((e) => SearchResult.fromJson(e)).toList() ?? [];
  }

  @override
  void dispose() {
    _client.close();
    super.dispose();
  }
}

class ApiException implements Exception {
  final String message;
  final int statusCode;
  ApiException(this.message, this.statusCode);
  @override
  String toString() => 'ApiException($statusCode): $message';
}

// ── List Response Wrappers ──

class TaskListResponse {
  final List<Task> tasks;
  final int total;
  TaskListResponse({required this.tasks, required this.total});
  factory TaskListResponse.fromJson(Map<String, dynamic> json) => TaskListResponse(
        tasks: (json['tasks'] as List?)?.map((e) => Task.fromJson(e)).toList() ?? [],
        total: json['total'] as int? ?? 0,
      );
}

class NoteListResponse {
  final List<Note> notes;
  final int total;
  NoteListResponse({required this.notes, required this.total});
  factory NoteListResponse.fromJson(Map<String, dynamic> json) => NoteListResponse(
        notes: (json['notes'] as List?)?.map((e) => Note.fromJson(e)).toList() ?? [],
        total: json['total'] as int? ?? 0,
      );
}

class ReminderListResponse {
  final List<Reminder> reminders;
  ReminderListResponse({required this.reminders});
  factory ReminderListResponse.fromJson(Map<String, dynamic> json) => ReminderListResponse(
        reminders: (json['reminders'] as List?)?.map((e) => Reminder.fromJson(e)).toList() ?? [],
      );
}

class MemoryListResponse {
  final List<MemoryEntry> entries;
  final int total;
  MemoryListResponse({required this.entries, required this.total});
  factory MemoryListResponse.fromJson(Map<String, dynamic> json) => MemoryListResponse(
        entries: (json['entries'] as List?)?.map((e) => MemoryEntry.fromJson(e)).toList() ?? [],
        total: json['total'] as int? ?? 0,
      );
}
