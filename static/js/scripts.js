document.addEventListener('DOMContentLoaded', function () {
    // 변수 선언 및 초기화
    const sendButton = document.getElementById('send-button');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const modelSelect = document.getElementById('model-select');
    const modelIntro = document.getElementById('model-intro');
    const questionsContainer = document.getElementById('questions-container'); // 예시 질문 컨테이너
    const toggleButton = document.getElementById('toggle-questions');
    const menuButton = document.getElementById('menu-button');
    const sideMenu = document.getElementById('side-menu');
    const overlay = document.createElement('div');
    overlay.id = 'overlay';
    document.body.appendChild(overlay);

    // Flask에서 전달된 모델 정보를 JavaScript 변수로 변환
    // const modelDescriptions는 HTML 템플릿에서 이미 정의됨

    // 기본 채팅 로고 HTML 설정
    const defaultChatLogoHTML = `
    <div class="chat-logo" id="chat-logo">
        <img src="/static/images/favicon.png" alt="GuminAI 로고">
        <p>안녕하세요! GuminAI입니다. 무엇을 도와드릴까요?</p>
    </div>
    `;
    chatMessages.innerHTML = defaultChatLogoHTML;

    // 봇 아바타 이미지 소스 초기화
    let botAvatarSrc = '/static/images/bot_avatar.png';

    // 모델 설명 및 봇 아바타 업데이트 함수
    function updateModelIntro() {
        const selectedModel = modelSelect.value;
        const description = modelDescriptions[selectedModel]?.description || '환영합니다! 기억 기능이 업데이트 되었습니다!';
        modelIntro.textContent = description;
        updateBotAvatar();
    }

    // 봇 아바타 이미지 업데이트 함수 (chat-logo 수정 부분 제거)
    function updateBotAvatar() {
        const selectedModel = modelSelect.value;
        const avatarImage = modelDescriptions[selectedModel]?.avatar_image || 'bot_avatar.png';
        botAvatarSrc = `/static/images/${avatarImage}`;
        // chat-logo 이미지는 수정하지 않음
    }

    // 초기 로드 시 모델 소개 업데이트
    updateModelIntro();

    // 모델 선택 변경 시 동작
    modelSelect.addEventListener('change', function () {
        updateModelIntro();
        // 모델 변경 시 대화 내역 초기화 API 호출
        fetch('/reset_conversation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        }).then(() => {
            // 채팅 창 초기화
            chatMessages.innerHTML = defaultChatLogoHTML;
            // 예시 질문 창 펼치기
            if (questionsContainer.classList.contains('collapsed')) {
                toggleQuestions();
            }
        }).catch(error => {
            console.error('대화 초기화 오류:', error);
        });
    });

    // 메시지 전송 이벤트 리스너 설정
    sendButton.addEventListener('click', sendMessage);

    // Enter 키로 메시지 전송 (모바일과 데스크탑 구분)
    chatInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isMobile()) {
                sendMessage();
            }
        }
    });

    chatInput.addEventListener('keydown', function (e) {
        if (isMobile()) {
            // 모바일에서는 Enter 키가 줄바꿈
            if (e.key === 'Enter' && !e.shiftKey) {
                // 기본 동작 허용 (줄바꿈)
            }
        } else {
            // 데스크탑에서는 Enter 키로 전송
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }
    });

    // 입력란의 높이를 자동으로 조절
    chatInput.addEventListener('input', function () {
        chatInput.style.height = 'auto';
        chatInput.style.height = chatInput.scrollHeight + 'px';
    });

    // 예시 질문 토글 버튼 이벤트 리스너
    toggleButton.addEventListener('click', toggleQuestions);

    // 예시 질문 클릭 시 메시지 전송
    questionsContainer.addEventListener('click', function (e) {
        if (e.target && e.target.matches('button.example-question')) {
            const question = e.target.textContent;
            chatInput.value = question;
            sendMessage();
        }
    });

    // 사이드 메뉴 열기 버튼 이벤트 리스너
    menuButton.addEventListener('click', function () {
        openSideMenu();
    });

    // 초기 예시 질문 로드
    refreshExampleQuestions();

    // 예시 질문을 새로 고치는 함수
    function refreshExampleQuestions() {
        fetch('/get_example_questions')
            .then(response => response.json())
            .then(data => {
                questionsContainer.innerHTML = '';
                data.example_questions.forEach(question => {
                    const button = document.createElement('button');
                    button.classList.add('example-question');
                    button.textContent = question;
                    questionsContainer.appendChild(button);
                });
            })
            .catch(error => {
                console.error('예시 질문 로드 오류:', error);
            });
    }

    // 예시 질문을 토글하는 함수
    function toggleQuestions() {
        questionsContainer.classList.toggle('collapsed');
        const isCollapsed = questionsContainer.classList.contains('collapsed');
        toggleButton.querySelector('img').src = isCollapsed ? '/static/images/arrow_down_icon.png' : '/static/images/arrow_up_icon.png';
    }

    // 사이드 메뉴 열기 함수
    function openSideMenu() {
        sideMenu.classList.add('open');
        overlay.classList.add('show');
        // 오버레이 클릭 시 사이드 메뉴 닫기
        overlay.addEventListener('click', closeSideMenu);
    }

    // 사이드 메뉴 닫기 함수
    function closeSideMenu() {
        sideMenu.classList.remove('open');
        overlay.classList.remove('show');
        // 이벤트 리스너 제거하여 메모리 누수 방지
        overlay.removeEventListener('click', closeSideMenu);
    }

    // 모바일 여부를 판단하는 함수
    function isMobile() {
        return /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    }

    // 메시지 전송 함수
    function sendMessage() {
        const message = chatInput.value.trim();
        if (message === '') return;

        // 사용자 메시지 추가
        addMessage('user', message);
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // 타이핑 인디케이터 표시
        showTypingIndicator();

        // 모델 소개 박스 숨기기
        if (modelIntro) {
            modelIntro.style.display = 'none';
        }

        // 모델 선택 드롭다운 및 전송 버튼 비활성화
        modelSelect.disabled = true;
        sendButton.disabled = true;
        sendButton.classList.add('disabled');

        // API 호출
        fetch('/chat_api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                model: modelSelect.value
            })
        })
        .then(response => response.json())
        .then(data => {
            hideTypingIndicator();
            // 모델 선택 드롭다운 및 전송 버튼 다시 활성화
            modelSelect.disabled = false;
            sendButton.disabled = false;
            sendButton.classList.remove('disabled');

            if (data.answer) {
                addMessage('bot', data.answer, function () {
                    if (data.reset_message) {
                        addMessage('system', data.reset_message);
                    }
                });
            } else {
                addMessage('bot', '죄송합니다. 응답을 가져올 수 없습니다.', function () {
                    if (data.reset_message) {
                        addMessage('system', data.reset_message);
                    }
                });
            }

            // 예시 질문 새로 고침
            refreshExampleQuestions();
        })
        .catch(error => {
            console.error('메시지 전송 오류:', error);
            hideTypingIndicator();
            // 모델 선택 드롭다운 및 전송 버튼 다시 활성화
            modelSelect.disabled = false;
            sendButton.disabled = false;
            sendButton.classList.remove('disabled');
            addMessage('bot', '죄송합니다. 오류가 발생했습니다.');
        });
    }

    // 메시지를 채팅창에 추가하는 함수
    function addMessage(sender, text, callback) {
        if (sender === 'system') {
            const messageElement = document.createElement('div');
            messageElement.classList.add('system-message');
            messageElement.textContent = text;
            chatMessages.appendChild(messageElement);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            return;
        }

        // 로고 숨기기
        const chatLogo = document.getElementById('chat-logo');
        if (chatLogo) {
            chatLogo.style.display = 'none';
        }

        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);

        const avatar = document.createElement('img');
        avatar.classList.add('avatar');
        avatar.src = sender === 'bot' ? botAvatarSrc : '/static/images/user_avatar.png'; // 'assistant'에서 'bot'으로 변경
        avatar.alt = sender === 'bot' ? 'Bot' : 'User'; // 'Assistant'에서 'Bot'으로 변경

        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');

        if (sender === 'bot') { // 'assistant'에서 'bot'으로 변경
            typeText(messageContent, text, 0, function () {
                if (callback) callback(); // 타이핑 완료 후 콜백 호출
            });
        } else {
            messageContent.textContent = text;
        }

        if (sender === 'user') {
            messageElement.appendChild(messageContent);
            messageElement.appendChild(avatar);
        } else {
            messageElement.appendChild(avatar);
            messageElement.appendChild(messageContent);
        }

        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // 메시지 복사 기능 추가 (터치 시 길게 누르면 복사)
        messageContent.addEventListener('touchstart', handleTouchStart, { passive: true });
        messageContent.addEventListener('touchend', handleTouchEnd);

        let touchTimer;

        function handleTouchStart(e) {
            touchTimer = setTimeout(() => {
                copyToClipboard(text);
            }, 1000); // 1초 이상 길게 누르면 복사
        }

        function handleTouchEnd(e) {
            if (touchTimer) {
                clearTimeout(touchTimer);
            }
        }
    }

    // 텍스트를 한 글자씩 타이핑하는 효과를 주는 함수
    function typeText(element, text, index = 0, callback) {
        if (index < text.length) {
            element.textContent += text.charAt(index);
            setTimeout(() => {
                typeText(element, text, index + 1, callback);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }, 30); // 타이핑 속도 조절 (밀리초)
        } else {
            if (callback) callback(); // 타이핑 완료 시 콜백 호출
        }
    }

    // 클립보드에 텍스트를 복사하는 함수
    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            console.log('복사 성공');
        }).catch(err => {
            console.error('복사 실패', err);
        });
    }

    // 타이핑 인디케이터를 표시하는 함수
    function showTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.classList.add('message', 'bot', 'typing-indicator'); // 'assistant'에서 'bot'으로 변경

        const avatar = document.createElement('img');
        avatar.classList.add('avatar');
        avatar.src = botAvatarSrc; // 선택된 모델에 따른 봇 아바타 이미지 사용
        avatar.alt = 'Bot'; // 'Assistant'에서 'Bot'으로 변경

        const dots = document.createElement('div');
        dots.classList.add('message-content');
        dots.innerHTML = `
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        `;

        typingIndicator.appendChild(avatar);
        typingIndicator.appendChild(dots);

        typingIndicator.id = 'typing-indicator';
        chatMessages.appendChild(typingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // 타이핑 인디케이터를 숨기는 함수
    function hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            chatMessages.removeChild(typingIndicator);
        }
    }
});
