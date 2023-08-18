import { useEffect, useState } from "react";

const MESSAGES = [
  { sender: "user", body: "Hey Riches! 👋🏼" },
  { sender: "bot", body: "Hello, Ask me anything about real estate 🤗" },
];

// eslint-disable-next-line react/prop-types
const Message = ({ bot, body, related = [] }) => {
  const [show, setShow] = useState(false);

  return (
    <div className={`flex ${!bot ? "justify-end" : ""}`}>
      <div
        className={`p-4 max-w-[80%] text-sm rounded-3xl shadow ${
          bot ? "rounded-bl-none bg-teal-900 text-white" : "rounded-br-none bg-white"
        }`}
      >
        <p>{body}</p>

        {related.length ? (
          <div className="mt-4">
            {!show ? (
              <button className="underline decoration-dotted font-medium" onClick={() => setShow(true)}>
                Show More
              </button>
            ) : (
              <div>
                <span className="font-medium underline uppercase">Similar Listings</span>
                {related.map((el, i) => (
                  <>
                    <ul key={i + "#"} className="list-none mt-2 mb-4 text-xs">
                      {Object.entries(el).map(([key, val]) => (
                        <li key={key} className="mb-2">
                          <span className="uppercase">{key}: </span>
                          {val}
                        </li>
                      ))}
                    </ul>
                    {i < related.length - 1 && <hr className="mb-4" />}
                  </>
                ))}
                <button className="underline decoration-dotted" onClick={() => setShow(false)}>
                  Hide All
                </button>
              </div>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default function App() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [messageList, setMessageList] = useState(MESSAGES);

  const handleSend = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      setMessageList((prev) => [...prev, { sender: "user", body: text }]);
      let response = await fetch(</BACKEND_URL>, {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: `user_query=${text}`,
      });
      response = await response.json();
      setMessageList((prev) => [
        ...prev,
        { sender: "bot", body: response.model_reply || "...", related: response.similar_listings },
      ]);
      setText("");
    } catch (error) {
      console.log(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const el = document.querySelector("#messages");
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messageList]);

  return (
    <div className="bg-gray-200 h-screen p-4">
      <section className="max-w-screen-md mx-auto flex flex-col justify-center h-full gap-8">
        <h1 className="text-3xl md:text-3xl font-black text-teal-900 text-center">
          Ask Riches about your real estate property needs! 🏘️
        </h1>
        {/*  */}
        <section className="shadow rounded-2xl">
          <header className="px-4 py-2.5 bg-teal-900 text-white rounded-t-2xl flex items-center gap-2 font-medium shadow">
            <img src="/bigSmile.svg" className="bg-gray-400 h-10 w-10 rounded-full" />
            Riches
          </header>
          {/*  */}
          <main id="messages" className="h-[45vh] bg-gray-100 px-4 py-4 flex flex-col gap-6 overflow-y-auto">
            {messageList.map((item, i) => (
              <Message key={i} bot={item.sender === "bot"} body={item.body} related={item.related} />
            ))}
          </main>
          {/*  */}
          <footer className="border-t bg-white p-4 rounded-b-2xl">
            <form onSubmit={handleSend}>
              <textarea
                rows="4"
                className="block w-full border rounded focus:outline-none focus:ring transition-all focus:ring-offset-2 focus:ring-teal-900/20 mb-4 text-sm p-2.5"
                placeholder="Type a message for Riches"
                autoFocus
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <button
                type="submit"
                disabled={loading || !text}
                className={`bg-teal-900 text-white text-sm font-medium px-6 py-3 rounded hover:bg-teal-950 focus:outline-none focus:ring transition-all focus:ring-offset-2 focus:ring-teal-900/20 ${
                  loading || !text ? "opacity-75 cursor-not-allowed" : "opacity-100"
                }`}
              >
                {loading ? "Sending..." : "Send Message"}
              </button>
            </form>
          </footer>
        </section>
      </section>
    </div>
  );
}
