import { useState, useRef, useEffect } from 'react';
import {
  Container,
  Row,
  Col,
  Form,
  Button,
  Spinner,
  Card,
  OverlayTrigger,
  Tooltip,
} from 'react-bootstrap';
import ReactMarkdown from 'react-markdown';
import { Lightbulb, Trash } from 'react-bootstrap-icons';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeFollowUpIndex, setActiveFollowUpIndex] = useState(null);
  const [history, setHistory] = useState([]);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  // simple demo session id – in a real app you may randomize this
  const sessionId = 'demo-session';

  // auto-scroll to bottom when history changes
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [history]);

  const handleSubmit = async (
    e,
    customQuery = null,
    isFollowUp = false,
    parentIndex = null
  ) => {
    e?.preventDefault?.();
    const userQuery = customQuery || query;
    if (!userQuery.trim()) return;

    setLoading(true);

    if (isFollowUp && parentIndex !== null) {
      setActiveFollowUpIndex(parentIndex);
    }

    try {
      const res = await fetch('http://localhost:5000/agentic-rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQuery, session_id: sessionId }),
      });

      const data = await res.json();

      // normalize suggested follow-ups: always an array
      const normalizeSuggested = (val) => {
        if (Array.isArray(val)) return val;
        if (typeof val === 'string') return [val];
        return [];
      };

      const newTurn = {
        userQuery,
        // backend returns "facts" as a single string – split into lines for display
        facts: Array.isArray(data.facts)
          ? data.facts
          : data.facts?.split('\n').filter((f) => f.trim() !== '') || [],
        // backend returns "answer"
        response: data.response || data.answer || '',
        suggestedQuestions: normalizeSuggested(data.suggestedQuestions),
      };

      // whether original or follow-up: just add a new turn
      setHistory((prev) => [...prev, newTurn]);

      setQuery('');
      inputRef.current?.focus();
    } catch (err) {
      console.error('Error calling /agentic-rag:', err);
    } finally {
      setLoading(false);
      setActiveFollowUpIndex(null);
    }
  };

  const resetChat = async () => {
    setHistory([]);
    setQuery('');
    try {
      await fetch(
        `http://localhost:5000/reset-session?session_id=${sessionId}`,
        {
          method: 'DELETE',
        }
      );
    } catch (err) {
      console.error('Failed to reset session:', err);
    }
  };

  return (
<Container fluid className="py-5">      <Row className="mb-4">
<Col xs={12} md={{ span: 8, offset: 2 }}>          <div className="d-flex justify-content-between align-items-center mb-3">
            <h2 className="mb-0">Agentic RAG with LangGraph</h2>
            <Button
              variant="outline-danger"
              size="sm"
              onClick={resetChat}
              disabled={loading}
              title="Reset Chat"
            >
              <Trash /> Reset
            </Button>
          </div>
          <Form onSubmit={handleSubmit}>
            <Form.Group className="mb-3">
              <Form.Label>Ask a question:</Form.Label>
              <Form.Control
                ref={inputRef}
                type="text"
                placeholder="e.g. How can I view and pay my bill?"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                required
                disabled={loading}
              />
            </Form.Group>
            <Button type="submit" variant="primary" disabled={loading}>
              {loading ? <Spinner animation="border" size="sm" /> : 'Submit'}
            </Button>
          </Form>
        </Col>
      </Row>

      {history.map((turn, index) => (
        <Row className="mb-4" key={index}>
          <Col md={{ span: 8, offset: 2 }}>
            {/* User query */}
            <Card className="mb-2">
              <Card.Header>User</Card.Header>
              <Card.Body>
                <p className="mb-0">{turn.userQuery}</p>
              </Card.Body>
            </Card>

            {/* Retrieved facts */}
            <Card className="mb-2">
              <Card.Header>Top Retrieved Facts</Card.Header>
              <Card.Body>
                {turn.facts.length > 0 ? (
                  <ul className="mb-0">
                    {turn.facts.map((fact, idx) => (
                      <li key={idx}>{fact}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="mb-0 text-muted">No facts retrieved.</p>
                )}
              </Card.Body>
            </Card>

            {/* Gemini response */}
            <Card className="mb-2">
              <Card.Header>Gemini Response</Card.Header>
              <Card.Body>
                {turn.response ? (
                  <ReactMarkdown>{turn.response}</ReactMarkdown>
                ) : (
                  <p className="mb-0 text-muted">No response generated.</p>
                )}
              </Card.Body>
            </Card>

            {/* Suggested follow-ups */}
            {Array.isArray(turn.suggestedQuestions) &&
              turn.suggestedQuestions.length > 0 && (
                <>
                  <div className="mb-2 d-flex align-items-center gap-2">
                    <Lightbulb className="text-warning" />
                    <strong>Suggested Follow-Ups:</strong>
                  </div>
                  <div className="mb-3 d-flex flex-wrap gap-2">
                    {turn.suggestedQuestions.map((q, i) => (
                      <OverlayTrigger
                        key={i}
                        placement="top"
                        overlay={
                          <Tooltip>Click to ask this follow-up</Tooltip>
                        }
                      >
                        <Button
                          variant="outline-success"
                          size="sm"
                          disabled={loading}
                          onClick={(e) => handleSubmit(e, q, true, index)}
                        >
                          {loading && activeFollowUpIndex === index ? (
                            <Spinner animation="border" size="sm" />
                          ) : (
                            q
                          )}
                        </Button>
                      </OverlayTrigger>
                    ))}
                  </div>
                </>
              )}
          </Col>
        </Row>
      ))}

      <div ref={bottomRef} />
    </Container>
  );
}

export default App;
