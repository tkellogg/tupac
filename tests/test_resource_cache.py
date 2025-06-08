import pytest
from collections import namedtuple
from tupac.resource_cache import ResourceCache, _process_tool_result


# Mock FastMCP TextContent
TextContent = namedtuple('TextContent', ['type', 'text'])
MockResult = namedtuple('MockResult', ['content'])


def test_cache_basic_operations():
    cache = ResourceCache()
    
    # Test empty cache
    assert not cache.contains("test://uri")
    
    # Test adding resource
    cache.add("test://uri", "Test Title", "text/plain", "Test content")
    assert cache.contains("test://uri")
    
    # Test adding same resource doesn't change flag
    cache._changed = False
    cache.add("test://uri", "Test Title", "text/plain", "Test content")
    assert not cache._changed


def test_cache_capacity_limit():
    cache = ResourceCache(capacity=2)
    
    cache.add("uri1", "Title 1", "text/plain", "Content 1")
    cache.add("uri2", "Title 2", "text/plain", "Content 2")
    cache.add("uri3", "Title 3", "text/plain", "Content 3")
    
    # First item should be evicted
    assert not cache.contains("uri1")
    assert cache.contains("uri2")
    assert cache.contains("uri3")


def test_consume_changed_blocks_generation():
    cache = ResourceCache()
    cache.add("test://uri", "Test Title", "text/plain", "Test content")
    
    blocks = cache.consume_changed_blocks()
    
    assert len(blocks) == 2
    assert '<resource uri="test://uri" title="Test Title" type="text/plain"/>' in blocks[0]
    assert '<resource uri="test://uri">Test content</resource>' in blocks[1]


def test_process_fastmcp_text_content():
    cache = ResourceCache()
    
    # Mock FastMCP result with TextContent
    text_content = TextContent(
        type='text',
        text='Regular text response'
    )
    result = MockResult(content=[text_content])
    
    processed = _process_tool_result(result, cache)
    assert processed == "Regular text response"


def test_process_search_results_with_resources():
    cache = ResourceCache()
    
    # Mock search result with resources
    search_json = '''{
        "results": [
            {
                "id": "https://example.com/article1",
                "title": "Test Article 1",
                "text": "This is the content of article 1",
                "type": "text/html"
            },
            {
                "id": "https://example.com/article2", 
                "title": "Test Article 2",
                "text": "This is the content of article 2"
            }
        ]
    }'''
    
    text_content = TextContent(type='text', text=search_json)
    result = MockResult(content=[text_content])
    
    processed = _process_tool_result(result, cache)
    
    # Should contain resource references
    assert '<resource uri="https://example.com/article1" title="Test Article 1" type="text/html"/>' in processed
    assert '<resource uri="https://example.com/article2" title="Test Article 2" type="text/plain"/>' in processed
    
    # Should contain content for first time
    assert '<resource_content uri="https://example.com/article1">This is the content of article 1</resource_content>' in processed
    assert '<resource_content uri="https://example.com/article2">This is the content of article 2</resource_content>' in processed
    
    # Check cache was populated
    assert cache.contains("https://example.com/article1")
    assert cache.contains("https://example.com/article2")


def test_process_cached_resources_blocks_content():
    cache = ResourceCache()
    
    # Pre-populate cache
    cache.add("https://example.com/article1", "Test Article 1", "text/html", "Cached content")
    
    # Same search result
    search_json = '''{
        "results": [
            {
                "id": "https://example.com/article1",
                "title": "Test Article 1", 
                "text": "This is the content of article 1",
                "type": "text/html"
            }
        ]
    }'''
    
    text_content = TextContent(type='text', text=search_json)
    result = MockResult(content=[text_content])
    
    processed = _process_tool_result(result, cache)
    
    # Should contain resource reference
    assert '<resource uri="https://example.com/article1" title="Test Article 1" type="text/html"/>' in processed
    
    # Should NOT contain content since it's cached
    assert '<resource_content uri="https://example.com/article1">' not in processed


def test_process_direct_dict_resource():
    cache = ResourceCache()
    
    # Direct dict with resource fields
    resource_dict = {
        "uri": "file://test.txt",
        "title": "Test File",
        "type": "text/plain",
        "text": "File content here"
    }
    
    processed = _process_tool_result(resource_dict, cache)
    
    assert '<resource uri="file://test.txt" title="Test File" type="text/plain"/>' in processed
    assert '<resource_content uri="file://test.txt">File content here</resource_content>' in processed
    assert cache.contains("file://test.txt")


def test_process_non_resource_content():
    cache = ResourceCache()
    
    # Regular non-resource content
    text_content = TextContent(type='text', text='Just a regular response')
    result = MockResult(content=[text_content])
    
    processed = _process_tool_result(result, cache)
    assert processed == "Just a regular response"
    
    # Cache should be empty
    assert len(cache.cache) == 0


def test_process_malformed_json():
    cache = ResourceCache()
    
    # Malformed JSON should not crash
    text_content = TextContent(type='text', text='{ invalid json }')
    result = MockResult(content=[text_content])
    
    processed = _process_tool_result(result, cache)
    assert processed == "{ invalid json }"


def test_process_mixed_content_types():
    cache = ResourceCache()
    
    # Mix of resource and non-resource content
    search_json = '''{
        "results": [
            {
                "id": "https://example.com/article1",
                "title": "Test Article",
                "text": "Article content"
            }
        ],
        "summary": "Found 1 result"
    }'''
    
    text_content = TextContent(type='text', text=search_json)
    result = MockResult(content=[text_content])
    
    processed = _process_tool_result(result, cache)
    
    # Should handle the resource
    assert '<resource uri="https://example.com/article1"' in processed
    assert cache.contains("https://example.com/article1")