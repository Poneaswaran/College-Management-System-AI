def test_ingest_success(client, internal_headers) -> None:
    response = client.post(
        "/ingest",
        headers=internal_headers,
        data={
            "material_id": "1",
            "subject_id": "10",
            "section_id": "20",
            "faculty_id": "30",
        },
        files={"file": ("notes.pdf", b"%PDF-1.4 test payload", "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "material:1"
    assert body["material_id"] == "1"


def test_ingest_rejects_invalid_extension(client, internal_headers) -> None:
    response = client.post(
        "/ingest",
        headers=internal_headers,
        data={
            "material_id": "1",
            "subject_id": "10",
            "section_id": "20",
            "faculty_id": "30",
        },
        files={"file": ("notes.exe", b"binary", "application/octet-stream")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "invalid_file_extension"


def test_query_success(client, internal_headers) -> None:
    response = client.post(
        "/query",
        headers=internal_headers,
        json={"message": "What is chapter 1 about?", "filters": {"material_id": 1}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert len(body["sources"]) == 1


def test_query_missing_filter(client, internal_headers) -> None:
    response = client.post(
        "/query",
        headers=internal_headers,
        json={"message": "hello", "filters": {}},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"


def test_query_no_context_fallback(client, internal_headers) -> None:
    response = client.post(
        "/query",
        headers=internal_headers,
        json={"message": "Any data?", "filters": {"material_id": 999}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert "could not find relevant context" in body["answer"].lower()


def test_delete_success(client, internal_headers) -> None:
    response = client.post(
        "/delete",
        headers=internal_headers,
        json={"vector_document_id": "material:1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] is True
    assert body["deleted_count"] == 2
    assert body["status"] == "deleted"


def test_delete_not_found(client, internal_headers) -> None:
    response = client.post(
        "/delete",
        headers=internal_headers,
        json={"vector_document_id": "material:999"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] is False
    assert body["deleted_count"] == 0
    assert body["status"] == "not_found"
